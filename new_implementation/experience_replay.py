import numpy as np
import torch

class ExperienceReplay():
    ##TODO: Image shape should be C,W,H
    def __init__(self,experience_size,observation_size,image_shape,reward_size,action_size,device):
        self.device=device
        #TODO: Observation size will only be used in symbolic envs
        self.observations = np.empty((experience_size,image_shape[0],image_shape[1],image_shape[2]),dtype=np.float32)
        self.actions = np.empty((experience_size,action_size),dtype=np.float32)
        self.rewards = np.empty((experience_size,reward_size),dtype=np.float32)
        self.non_terminals = np.empty((experience_size,1),dtype=np.float32)
        self.idx, self.steps, self.episodes = 0, 0, 0 
        self.full =False
    def append(self,observation,reward,action,done ):
        self.observations[self.idx] = observation## Needs postprocessing
        self.rewards[self.idx]=reward
        self.actions[self.idx]=action
        self.non_terminals[self.idx]=not done

