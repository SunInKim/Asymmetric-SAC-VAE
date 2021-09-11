import os
import argparse
from datetime import datetime
from agent import AsymSacVae
from env import Panda

def run():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_type', type=str, default='AsymSacVae')
	parser.add_argument('--env_id', type=str, default='tidy-up')
	parser.add_argument('--memory_size', type=int, default=40000)
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--num_task', type=int, default=2)
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--init_latent', action='store_true')
	parser.add_argument('--GUI', default='DIRECT')

	args = parser.parse_args()

	# Configs which are constant across all tasks.
	configs = {
		'env_type': args.env_type,
		'num_steps': 3000000,
		'initial_latent_steps': 30000,
		'batch_size':64,
		'latent_batch_size': 32,
		'beta': 4,
		'lr': 0.0002,
		'latent_lr': 0.0001,
		'feature_dim': 256,
		'latent_dim': 256,
		'hidden_units': [512, 512, 512],
		'memory_size': args.memory_size,
		'gamma': 0.99,
		'target_update_interval': 1,
		'tau': 0.005,
		'entropy_tuning': True,
		'ent_coef': 0.2,  # It's ignored when entropy_tuning=True.
		'leaky_slope': 0.2,
		'grad_clip': None,
		'updates_per_step': 1,
		'start_steps': 30000,
		'training_log_interval': 10,
		'learning_log_interval': 100,
		'eval_interval': 5000,
		'init_latent': args.init_latent,
		'cuda': args.cuda,
		'seed': args.seed
	}

	env = Panda(args)

	dir_name = args.env_id
	log_dir = os.path.join(
		'logs', args.env_type, dir_name,
		f'asymsacvae-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}')

	agent = AsymSacVae(env=env, log_dir=log_dir, **configs)
	# If train
	# agent.run()

	# If Test
	agent.test_episode()

if __name__ == '__main__':
	run()
