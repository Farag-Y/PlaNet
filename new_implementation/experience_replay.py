import numpy as np
from typing import List
import torch
class ExperienceReplay():
    ##TODO: Image shape should be C,W,H
    def __init__(self,experience_size,observation_size,image_shape,action_size,device):
        self.device=device
        #TODO: Observation size will only be used in symbolic envs
        self.observations = np.empty((experience_size,image_shape[0],image_shape[1],image_shape[2]),dtype=np.float32)
        self.actions = np.empty((experience_size,action_size),dtype=np.float32)
        self.rewards = np.empty((experience_size,),dtype=np.float32)
        self.non_terminals = np.empty((experience_size,1),dtype=np.float32)
        self.idx, self.steps, self.episodes = 0, 0, 0 
        self.full =False
        self.size=experience_size
    def append(self,observation,reward,action,done ):
        self.observations[self.idx] = observation## Needs postprocessing
        self.rewards[self.idx]=reward
        self.actions[self.idx]=action
        self.non_terminals[self.idx]=not done
        self.idx = (self.idx+1)%self.size
        self.full = self.full or self.idx==0
        self.steps+=1 # TODO: Can't we use idx instead here
        self.episodes +=(1 if not done else 0 ) # Also same question here
    def _get_indexes(self,batch_size:int,batch_length:int)->List[int]:
        valid_idx=False
        batches = []
        for i in range(batch_size):
            valid_idx = False
            while not valid_idx:
                max_size = self.idx-batch_length if not self.full else self.size
                idx = np.random.randint(0,max_size)
                idxs = np.arange(idx,idx+batch_length)%self.size
                valid_idx = self.idx not in idxs[1:]
            batches.append(idxs)
        return batches
    

    def _get_batch(self,idxs,batch_size,batch_length):
        obs = torch.as_tensor(self.observations[idxs]).to(self.device)
        acts = torch.as_tensor(self.actions[idxs]).to(self.device)
        rewards = torch.as_tensor(self.rewards[idxs]).to(self.device)
        non_terminals = torch.as_tensor(self.non_terminals[idxs]).to(self.device)
        return obs,acts,rewards,non_terminals ##TODO: This function needs post processing of observations + make sure that the format is correct.
    def sample(self,batch_size,batch_length):
        batch_idxs = self._get_indexes(batch_size,batch_length)
        batches = self._get_batch(batch_idxs,batch_size,batch_length)
        return batches
