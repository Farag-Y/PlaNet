import gymnasium as gym
from gymnasium.spaces import Box
from typing import cast
from experience_replay import ExperienceReplay



def collect_observations(env: gym.Env,seed_episodes,experience_size,image_shape,device)->ExperienceReplay:
    obs_space = cast(Box, env.observation_space)
    act_space = cast(Box, env.action_space)
    experience_replay = ExperienceReplay(experience_size, obs_space.shape[0], image_shape, act_space.shape[0], device)


    for i in range(seed_episodes):
        terminated = False
        observation, info = env.reset()
        while not terminated:
            action = env.action_space.sample() 
            observation, reward, terminated, truncated, _ = env.step(action)
            # total_reward += reward
            # episode_over = terminated or truncated
            experience_replay.append(observation,reward,action,terminated or truncated)
    env.close()
    return experience_replay

def main():
    env = gym.make("Pendulum-v1", render_mode="None")
    seed_episodes=10
    experience_size=10000
    image_shape=[3,64,64]
    device = "cpu"
    collect_observations(env,seed_episodes=seed_episodes,experience_size=experience_size,image_shape=image_shape,device=device)

if __name__ == "__main__":
    main()