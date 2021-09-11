from collections import deque
import numpy as np
import torch

class Memory(dict):
    ''' LazyMemory is memory-efficient but time-inefficient. '''
    keys = ['state', 'action', 'reward', 'next_state', 'done', 'task', 'gt_state', 'next_gt_state', 'next_task']

    def __init__(self, capacity, observation_shape,
     seg_shape, dep_shape, action_shape, gt_state_shape, device):
        super(Memory, self).__init__()
        self.capacity = int(capacity)
        self.observation_shape = (observation_shape[0]+dep_shape[0]+seg_shape[0], 256, 256)
        self.action_shape = action_shape
        self.gt_state_shape = gt_state_shape
        self.device = device
        self.reset()

    def reset(self):
        self.is_set_init = False
        self._p = 0
        self._n = 0
        for key in self.keys:
            self[key] = [None] * self.capacity

    def append(self, state, action, reward, next_state, done, task, gt_state, next_gt_state, next_task):
        self['state'][self._p] = state
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['next_state'][self._p] = next_state
        self['done'][self._p] = done
        self['task'][self._p] = task
        self['gt_state'][self._p] = gt_state
        self['next_gt_state'][self._p] = next_gt_state
        self['next_task'][self._p] = next_task

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample_latent(self, batch_size):

        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)


        for i, index in enumerate(indices):
            states[i, ...] = self['state'][index]

        states = torch.ByteTensor(states).to(self.device).float()/127.5 -1.0
        
        return states

    def sample_sac(self, batch_size):

        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        actions = np.empty((batch_size, *self.action_shape), dtype=np.float32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        next_states = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        tasks = np.empty((batch_size, 10), dtype=np.float32)
        gt_states = np.empty((batch_size, *self.gt_state_shape), dtype=np.float32)
        next_gt_states = np.empty((batch_size, *self.gt_state_shape), dtype=np.float32)
        next_tasks = np.empty((batch_size, 10), dtype=np.float32)

        for i, index in enumerate(indices):
            # Convert LazeFrames into np.ndarray here.
            states[i, ...] = self['state'][index]
            actions[i, ...] = self['action'][index]
            rewards[i, ...] = self['reward'][index]
            next_states[i, ...] = self['next_state'][index]
            tasks[i, ...] = self['task'][index]
            gt_states[i, ...] = self['gt_state'][index]
            next_gt_states[i, ...] = self['gt_state'][index]
            next_tasks[i, ...] = self['next_task'][index]

        states = torch.ByteTensor(states).to(self.device).float()/127.5 -1.0
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.ByteTensor(next_states).to(self.device).float()/127.5 -1.0
        tasks = torch.FloatTensor(tasks).to(self.device)
        gt_states = torch.FloatTensor(gt_states).to(self.device)
        next_gt_states = torch.FloatTensor(next_gt_states).to(self.device)
        next_tasks = torch.FloatTensor(next_tasks).to(self.device)

        return states, actions, rewards, next_states, tasks, gt_states, next_gt_states, next_tasks

    def __len__(self):
        return self._n
