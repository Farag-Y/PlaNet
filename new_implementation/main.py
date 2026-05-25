from env_wrapper import Env
from experience_replay import ExperienceReplay


def collect_observations(env, seed_episodes, experience_replay: ExperienceReplay) -> ExperienceReplay:
    for _ in range(seed_episodes):
        observation = env.reset()          # tensor (1, 3, 64, 64), already preprocessed
        done = False
        while not done:
            action = env.sample_random_action()           # torch tensor
            observation, reward, done = env.step(action)  # (tensor, float, bool)
            experience_replay.append(observation, reward, action, done)
    env.close()
    return experience_replay


def main():
    env = Env("Pendulum-v1", seed=0, max_episode_length=1000, action_repeat=2, bit_depth=5)
    seed_episodes = 10
    experience_size = 10000
    device = "cpu"
    experience_replay = ExperienceReplay(
        experience_size,
        observation_size=0,                      # 0 = visual env (symbolic not used here)
        image_shape=list(env.observation_size),  # (3, 64, 64) from wrapper
        action_size=env.action_size,
        device=device,
    )
    #TODO: Clean up what is above a bit.
    transition_model = TransitionModel(args.belief_size, args.state_size, env.action_size, args.hidden_size, args.embedding_size, args.activation_function).to(device=args.device)

    collect_observations(env, seed_episodes=seed_episodes, experience_replay=experience_replay)
    #TODO: Prepare the main models, and initalize
    #TODO:

if __name__ == "__main__":
    main()
