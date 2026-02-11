import argparse
import pathlib

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import humanoid_bench
from .env import ROBOTS, TASKS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="HumanoidBench environment test")
    parser.add_argument("--env", help="e.g. h1-walk-v0")
    parser.add_argument("--keyframe", default=None)
    parser.add_argument("--policy_path", default=None)
    parser.add_argument("--mean_path", default=None)
    parser.add_argument("--var_path", default=None)
    parser.add_argument("--policy_type", default=None)
    parser.add_argument("--blocked_hands", default="False")
    parser.add_argument("--small_obs", default="False")
    parser.add_argument("--obs_wrapper", default="False")
    parser.add_argument("--sensors", default="")
    parser.add_argument("--render_mode", default="rgb_array")  # "human" or "rgb_array".
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--save_video", action="store_true", help="Save video of all episodes")
    # NOTE: to get (nicer) 'human' rendering to work, you need to fix the compatibility issue between mujoco>3.0 and gymnasium: https://github.com/Farama-Foundation/Gymnasium/issues/749
    args = parser.parse_args()

    kwargs = vars(args).copy()
    kwargs.pop("env")
    kwargs.pop("render_mode")
    if kwargs["keyframe"] is None:
        kwargs.pop("keyframe")
    print(f"arguments: {kwargs}")

    # Test offscreen rendering
    print(f"Test offscreen mode...")
    env = gym.make(args.env, render_mode="rgb_array", **kwargs)
    ob, _ = env.reset()
    if isinstance(ob, dict):
        print(f"ob_space = {env.observation_space}")
        print(f"ob = ")
        for k, v in ob.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
    print(f"ac_space = {env.action_space.shape}")

    img = env.render()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("test_env_img.png", rgb_img)

    # Test online rendering with interactive viewer
    print(f"Test onscreen mode...")
    env = gym.make(args.env, render_mode=args.render_mode, **kwargs)
    ob, _ = env.reset()
    if isinstance(ob, dict):
        print(f"ob_space = {env.observation_space}")
        print(f"ob = ")
        for k, v in ob.items():
            print(f"  {k}: {v.shape}")
            assert (
                v.shape == env.observation_space.spaces[k].shape
            ), f"{v.shape} != {env.observation_space.spaces[k].shape}"
        assert ob.keys() == env.observation_space.spaces.keys()
    else:
        print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
        assert env.observation_space.shape == ob.shape
    print(f"ac_space = {env.action_space.shape}")
    # print("observation:", ob)
    
    video_writer = None
    if args.save_video:
        img = env.render()
        video_filename = f"video_{args.env.replace('-', '_')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, 
                                       (img.shape[1], img.shape[0]))
        print(f"Recording video to: {video_filename}")

    episode_returns = []
    episode_lengths = []
    
    print(f"\nRunning {args.num_episodes} episodes with random policy...")
    
    for episode in range(args.num_episodes):
        ob, _ = env.reset()
        episode_return = 0
        episode_length = 0
            
    # env.render()
    # ret = 0
        while True:
            action = env.action_space.sample()
            ob, rew, terminated, truncated, info = env.step(action)
            episode_return += rew
            episode_length += 1
            
            img = env.render()
            # ret += rew
            
            if video_writer is not None:
                video_writer.write(img[:, :, ::-1])

            if args.render_mode == "rgb_array":
                cv2.imshow("test_env", img[:, :, ::-1])
                cv2.waitKey(1)

            if terminated or truncated:
                # ret = 0
                # env.reset()
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                print(f"Episode {episode + 1}/{args.num_episodes}: Return = {episode_return:.2f}, Length = {episode_length}")
                break
    env.close()
    cv2.destroyAllWindows()
    
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {video_filename}")

    
    # Calculate statistics
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    mean_length = np.mean(episode_lengths)
    
    print(f"\n{'='*50}")
    print(f"Results for {args.env} (Random Policy):")
    print(f"{'='*50}")
    print(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Min Return: {np.min(episode_returns):.2f}")
    print(f"Max Return: {np.max(episode_returns):.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot episode returns over time
    ax.plot(range(1, len(episode_returns) + 1), episode_returns, 'b-o', alpha=0.7)
    ax.axhline(y=mean_return, color='r', linestyle='--', label=f'Mean: {mean_return:.2f}')
    ax.fill_between(range(1, len(episode_returns) + 1), 
                    mean_return - std_return, 
                    mean_return + std_return, 
                    alpha=0.2, color='r', label=f'±1 Std: {std_return:.2f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title(f'Episode Returns - {args.env}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
        
    # Save plot
    plot_filename = f"performance_{args.env.replace('-', '_')}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_filename}")
    
    plt.show()