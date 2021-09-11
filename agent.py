import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from memory import Memory
from network import GaussianPolicy, TwinnedQNetwork, LatentNetwork
from utils import grad_false, hard_update, soft_update, update_params, RunningMeanStats


class AsymSacVae:
    def __init__(self, 
        env, 
        log_dir, 
        env_type='dm_control', 
        num_steps=3000000, 
        initial_latent_steps=100000, 
        batch_size=256, 
        latent_batch_size=32, 
        beta=4, 
        lr=0.0003, 
        latent_lr=0.0001, 
        feature_dim=256, 
        latent_dim=256, 
        hidden_units=[256, 256], 
        memory_size=1e5, 
        gamma=0.99, 
        target_update_interval=1, 
        tau=0.005, 
        entropy_tuning=True, 
        ent_coef=0.2, 
        leaky_slope=0.2, 
        grad_clip=None, 
        updates_per_step=1, 
        start_steps=10000, 
        training_log_interval=10, 
        learning_log_interval=100, 
        eval_interval=20000, 
        init_latent=False, 
        cuda=True, 
        seed=0):

        # Data type setting
        self.env = env
        self.observation_shape = self.env.observation_space.shape
        self.seg_shape = self.env.seg_space.shape
        self.dep_shape = self.env.dep_space.shape
        self.action_shape = self.env.action_space.shape
        self.hybrid_shape = self.env.hybrid_space.shape
        self.gt_state_shape = self.env.gt_state_space.shape         
        self.max_step = env._max_episode_steps
     
        # Seed 
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        # Device setting
        self.device = torch.device(
            "cuda:0" if cuda and torch.cuda.is_available() else "cpu")
        print(self.device)

        # Network setting
        self.policy = GaussianPolicy(
            feature_dim + self.hybrid_shape[0],
            self.action_shape[0], hidden_units).to(self.device)
        self.critic = TwinnedQNetwork(self.gt_state_shape[0], self.action_shape[0], hidden_units
            ).to(self.device)
        self.critic_target = TwinnedQNetwork(self.gt_state_shape[0], self.action_shape[0], hidden_units
            ).to(self.device).eval()
        self.latent = LatentNetwork(
            self.observation_shape[0]+self.seg_shape[0]+self.dep_shape[0], self.action_shape, feature_dim,
            latent_dim, hidden_units, leaky_slope
            ).to(self.device)
        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)
        
        # Policy is updated without the encoder.
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)
        self.latent_optim = Adam(self.latent.parameters(), lr=latent_lr)

        # Entrophy tuning
        if entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(
                torch.Tensor(self.action_shape)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = torch.tensor(ent_coef).to(self.device)

        # Replay buffer
        self.memory = Memory(
            memory_size, self.observation_shape, self.seg_shape, self.dep_shape, self.action_shape, self.gt_state_shape, self.device)

        # Directory setting
        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_rewards = RunningMeanStats(training_log_interval)

        # Loss setting
        self.mask_criterion = nn.L1Loss().to(self.device)        

        # Hyper-param setting
        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.initial_latent_steps = initial_latent_steps
        self.num_steps = num_steps
        self.init_latent = init_latent
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size
        self.latent_batch_size = latent_batch_size
        self.start_steps = start_steps
        self.gamma = gamma
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.training_log_interval = training_log_interval
        self.learning_log_interval = learning_log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and\
            self.steps >= self.start_steps

    def deque_to_batch(self, state, task):
        # Convert deques to batched tensor.
        state = np.array(state, dtype=np.uint8)
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 127.5 -1.0
        task = torch.FloatTensor(
            task).unsqueeze(0).to(self.device)
        with torch.no_grad():

            state = self.augmentation(state)
            feature, _ = self.latent.encoder(state)
            feature = feature.view(1, -1)

        feature_state = torch.cat([feature, task], dim=-1)
        return feature_state

    def explore(self, state, task):
        feature_state = self.deque_to_batch(state, task)
        
        with torch.no_grad():
            action, _, _ = self.policy.sample(feature_state)
        return action.cpu().numpy().reshape(-1)

    def execution(self, state, task):
        feature_state = self.deque_to_batch(state, task)
        
        with torch.no_grad():
            _, _, action = self.policy.sample(feature_state)
        return action.cpu().numpy().reshape(-1)

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        ori_img, task, seg_img, dep_img, gt_state = self.env.reset()

        state = np.concatenate([ori_img,seg_img,dep_img], axis=0)

        num_obj = self.env.get_num_obj()

        epi_step = 0
        while not done:

            action = self.explore(state, task)

            ori_img, reward, done, _ , next_task, seg_img, dep_img, next_gt_state = self.env.step(action)
            next_state = np.concatenate([ori_img,seg_img,dep_img], axis=0)

            self.steps += 1
            episode_steps += 1
            episode_reward += reward
            epi_step += 1

            self.memory.append(state, action, reward, next_state, done, task, gt_state, next_gt_state, next_task)

            if self.is_update():
            #     # First, train the latent model only.
                if self.learning_steps < self.initial_latent_steps and self.init_latent:
                    print('-'*60)
                    print('Learning the latent model only...')
                    for _ in range(self.initial_latent_steps):
                        self.learning_steps += 1
                        if self.learning_steps % 1000 == 0:
                            print(self.learning_steps)
                        self.learn_latent()
                    print('Finish learning the latent model.')
                    print('-'*60)

                for _ in range(self.updates_per_step):
                    self.learn()

                if self.steps % self.eval_interval == 0:
                    self.evaluate()
                    self.save_models()
            if epi_step == self.max_step:
                break
            state = next_state
            gt_state = next_gt_state
            task = next_task

        # We log running mean of training rewards.
        self.train_rewards.append(episode_reward)

        if self.episodes % self.training_log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_rewards.get(), self.steps)

        print(f'episode: {self.episodes:<4} '
              f'Task ID: {task[:2]}  '
              f'NUM OBJ: {num_obj}  '
              f'episode steps: {episode_steps:<4} '
              f'reward: {episode_reward:<3.2f}  '
              f'total steps: {self.steps:<7}')

    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # Update the latent model.
        self.learn_latent()
        # Update policy and critic.
        self.learn_sac()

    def learn_latent(self):
        images =\
            self.memory.sample_latent(self.latent_batch_size)

        latent_loss = self.calc_latent_loss(images)
        update_params(
            self.latent_optim, self.latent, latent_loss, self.grad_clip)

        if self.learning_steps % self.learning_log_interval == 0:
            self.writer.add_scalar(
                'loss/latent_loss', latent_loss.detach().item(),
                self.learning_steps)

    def learn_sac(self):
        images, actions, rewards, next_images, tasks, gt_states, next_gt_states, next_tasks =\
            self.memory.sample_sac(self.batch_size)

        # NOTE: Don't update the encoder part of the policy here.
        with torch.no_grad():

            images  = self.augmentation(images)
            features,_ = self.latent.encoder(images)

            next_images  = self.augmentation(next_images)
            next_features,_ = self.latent.encoder(next_images)

        states = torch.cat((features, tasks), dim=1)
        next_states = torch.cat((next_features, next_tasks), dim=1)

        q1_loss, q2_loss = self.calc_critic_loss(
            states, gt_states, actions, rewards, next_states, next_gt_states)
        policy_loss, entropies = self.calc_policy_loss(gt_states, states)

        update_params(
            self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(
            self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
        else:
            entropy_loss = 0.

        if self.learning_steps % self.learning_log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)

    def calc_latent_loss(self, images):

        images = self.augmentation(images)
        features, distribution = self.latent.encoder(images)

        sample = distribution.rsample()

        mu = distribution.loc
        logvar = distribution.scale

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Log likelihood loss of generated observations.
        img_dists = self.latent.decoder(
            sample)

        log_likelihood_loss = img_dists.log_prob(
            images).mean(dim=0).sum()

        reconst_error = self.mask_criterion(img_dists.loc,images)

        latent_loss =\
            self.beta*KLD - log_likelihood_loss + 100*reconst_error

        if self.learning_steps % self.learning_log_interval == 0:
            reconst_error = (
                images - img_dists.loc
                ).pow(2).mean(dim=(0, 1)).sum().item()

            self.writer.add_scalar(
                'stats/reconst_error', reconst_error, self.learning_steps)
      
        return latent_loss

    def calc_critic_loss(self, states, gt_states, actions, rewards, next_states, next_gt_states):
        # Q(z(t), a(t))
        curr_q1, curr_q2 = self.critic(gt_states, actions)
        # E[Q(z(t+1), a(t+1)) + alpha * H(pi)]
        with torch.no_grad():
            next_actions, next_entropies, _ =\
                self.policy.sample(states)
            next_q1, next_q2 = self.critic_target(next_gt_states, next_actions)
            next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies
        # r(t) + gamma * E[Q(z(t+1), a(t+1)) + alpha * H(pi)]
        target_q = rewards + self.gamma * next_q

        # Critic losses are mean squared TD errors.
        q1_loss = 0.5 * torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = 0.5 * torch.mean((curr_q2 - target_q).pow(2))

        if self.learning_steps % self.learning_log_interval == 0:
            mean_q1 = curr_q1.detach().mean().item()
            mean_q2 = curr_q2.detach().mean().item()
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)

        return q1_loss, q2_loss

    def calc_policy_loss(self, gt_states, states):
        # Re-sample actions to calculate expectations of Q.
        sampled_actions, entropies, _ = self.policy.sample(states)
        # E[Q(z(t), a(t))]
        q1, q2 = self.critic(gt_states, sampled_actions)
        q = torch.min(q1, q2)

        # Policy objective is maximization of (Q + alpha * entropy).
        policy_loss = torch.mean((- q - self.alpha * entropies))

        return policy_loss, entropies

    def calc_entropy_loss(self, entropies):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies).detach())
        return entropy_loss

    def augmentation(self, cat_img, out=240):
        
        n, c, h, w = cat_img.shape
        crop_max = h - out + 1
        w1 = np.random.randint(0, crop_max, n)
        h1 = np.random.randint(0, crop_max, n)
        cropped_cat = torch.zeros((n, c, out, out), dtype=cat_img.dtype).to(self.device)

        for i, (img, w11, h11) in enumerate(zip(cat_img, w1, h1)):
            cropped_cat[i] = img[:, h11:h11 + out, w11:w11 + out]


        return cropped_cat

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            ori_img, task, seg_img, dep_img, gt_state = self.env.reset()
            state = np.concatenate([ori_img,seg_img,dep_img],axis=0)
            episode_reward = 0.
            epi_step = 0
            done = False

            while not done:
                action = self.explore(state, task)
                ori_img, reward, done, _, next_task, seg_img, dep_img, next_gt_state = self.env.step(action)
                next_state = np.concatenate([ori_img,seg_img,dep_img],axis=0)
                episode_reward += reward
                epi_step += 1

                state = next_state
                gt_state = next_gt_state
                task = next_task

                if epi_step == self.max_step:
                    break

            returns[i] = episode_reward

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        print('-' * 60)
        print(f'environment steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f} +/- {std_return:<5.1f}')
        print('-' * 60)

    

    def test_episode(self):
        self.load_models()

        episodes = 10000
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            ori_img, task, seg_img, dep_img, gt_state = self.env.reset()
            state = np.concatenate([ori_img,seg_img,dep_img],axis=0)
            episode_reward = 0.
            epi_step = 0
            done = False
            num_obj = self.env.get_num_obj()

            while not done:
                action = self.execution(state, task)
                ori_img, reward, done, _, next_task, seg_img, dep_img, next_gt_state = self.env.step(action)
                next_state = np.concatenate([ori_img,seg_img,dep_img],axis=0)
                episode_reward += reward
                epi_step += 1

                state = next_state
                gt_state = next_gt_state
                task = next_task

                if epi_step == self.max_step:
                    break

            returns[i] = episode_reward

            print(f'episode: {episodes:<4} '
              f'Task ID: {task[:2]}  '
              f'NUM OBJ: {num_obj}  '
              f'episode steps: {epi_step:<4} '
              f'reward: {episode_reward:<3.2f}  ')
      
    def save_models(self):
        self.latent.save(os.path.join(self.model_dir, 'latent.pth'))
        self.policy.save(os.path.join(self.model_dir, 'policy.pth'))
        self.critic.save(os.path.join(self.model_dir, 'critic.pth'))
        self.critic_target.save(os.path.join(self.model_dir, 'critic_target.pth'))

    def load_models(self):
        model_dir = "model/"
        self.policy.load(os.path.join(model_dir, 'policy.pth'))
        self.critic.load(os.path.join(model_dir, 'critic.pth'))
        self.latent.load(os.path.join(model_dir, 'latent.pth'))
        self.critic_target.load(os.path.join(model_dir, 'critic_target.pth'))

    def __del__(self):
        self.writer.close()
        self.env.close()
