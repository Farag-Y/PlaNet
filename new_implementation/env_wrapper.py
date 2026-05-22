import abc
import cv2
import numpy as np
import torch


CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2}


class BaseEnv(abc.ABC):
    def preprocess_observation_(self, observation, bit_depth):
        observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
        observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))

    def postprocess_observation(self, observation, bit_depth):
        return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)

    def _images_to_observation(self, images, bit_depth):
        images = torch.tensor(cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)
        self.preprocess_observation_(images, bit_depth)
        return images.unsqueeze(dim=0)

    @abc.abstractmethod
    def reset(self): ...

    @abc.abstractmethod
    def step(self, action): ...

    @abc.abstractmethod
    def render(self): ...

    @abc.abstractmethod
    def close(self): ...

    @property
    @abc.abstractmethod
    def observation_size(self): ...

    @property
    @abc.abstractmethod
    def action_size(self): ...

    @property
    @abc.abstractmethod
    def action_range(self): ...

    @abc.abstractmethod
    def sample_random_action(self): ...


class ControlSuiteEnv(BaseEnv):
    def __init__(self, env, seed, max_episode_length, action_repeat, bit_depth):
        from dm_control import suite
        domain, task = env.split('-')
        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
            print('Using action repeat %d; recommended action repeat for domain is %d' % (action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain]))
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0
        self._env.reset()
        return self._images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for _ in range(self.action_repeat):
            state = self._env.step(action)
            reward += state.reward
            self.t += 1
            done = state.last() or self.t == self.max_episode_length
            if done:
                break
        observation = self._images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)
        return observation, reward, done

    def render(self):
        cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self._env.close()

    @property
    def observation_size(self):
        return (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]

    @property
    def action_range(self):
        return float(self._env.action_spec().minimum[0]), float(self._env.action_spec().maximum[0])

    def sample_random_action(self):
        spec = self._env.action_spec()
        return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))


class GymEnv(BaseEnv):
    def __init__(self, env, seed, max_episode_length, action_repeat, bit_depth):
        import gym
        import logging
        gym.logger.set_level(logging.ERROR)
        self._env = gym.make(env, render_mode='rgb_array')
        self._seed = seed
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0
        self._env.reset(seed=self._seed)
        self._seed = None
        return self._images_to_observation(self._env.render(), self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for _ in range(self.action_repeat):
            _, reward_k, terminated, truncated, _ = self._env.step(action)
            reward += reward_k
            self.t += 1
            done = terminated or truncated or self.t == self.max_episode_length
            if done:
                break
        observation = self._images_to_observation(self._env.render(), self.bit_depth)
        return observation, reward, done

    def render(self):
        frame = self._env.render()
        if frame is not None:
            cv2.imshow('screen', frame[:, :, ::-1])
            cv2.waitKey(1)

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    @property
    def action_range(self):
        return float(self._env.action_space.low[0]), float(self._env.action_space.high[0])

    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())
