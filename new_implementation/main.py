import gymnasium as gym

import env



def _collect_observations(env: gym.Env):
    observation, info = env.reset()

    episode_over = False
    total_reward = 0

    while not episode_over:
        action = env.action_space.sample() 
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()


def __main__():
    env = gym.make("Pendulum-v1", render_mode="human")
