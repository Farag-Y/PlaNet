import argparse
import sys

import cv2
import numpy as np
import torch

from constants import ENV_KEY_MAPS, ENV_HELP
from env import GymEnv, GYM_ENVS


def main():
    parser = argparse.ArgumentParser(description='Play a gym environment interactively')
    parser.add_argument('--env', default='Pendulum-v1', choices=GYM_ENVS)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(f'Env: {args.env}  |  Controls: {ENV_HELP[args.env]}  |  q: quit')

    key_map = ENV_KEY_MAPS[args.env]
    env = GymEnv(args.env, symbolic=True, seed=args.seed,
                 max_episode_length=1000, action_repeat=1, bit_depth=5)
    zero_action = np.zeros(env.action_size, dtype=np.float32)

    for ep in range(args.episodes):
        env.reset()
        total_reward = 0.0
        done = False
        while not done:
            frame = env._env.render()
            cv2.imshow(args.env, frame[:, :, ::-1])
            raw = cv2.waitKey(50)
            key = (raw & 0xFF) if raw != -1 else -1
            if key == ord('q'):
                cv2.destroyAllWindows()
                env.close()
                sys.exit()
            action_np = key_map.get(key, zero_action).astype(np.float32)
            _, reward, done = env.step(torch.tensor(action_np))
            total_reward += reward
        print(f'Episode {ep + 1}: total reward = {total_reward:.2f}')

    cv2.destroyAllWindows()
    env.close()


if __name__ == '__main__':
    main()
