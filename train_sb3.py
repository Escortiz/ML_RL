"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import argparse
import torch
import gym
from stable_baselines3 import SAC, PPO
from env.custom_hopper import *


def get_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA non disponibile, uso CPU.")
        return "cpu"
    return device


def main():
    parser = argparse.ArgumentParser(
        description="Addestramento di politiche con SAC o PPO su CustomHopper."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["SAC", "PPO"],
        default="SAC",
        help="Algoritmo di RL da utilizzare.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Numero di passi di addestramento.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Dispositivo per l'addestramento.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="trained_model",
        help="Percorso per salvare il modello addestrato.",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    train_env = gym.make("CustomHopper-source-v0")

    print("State space:", train_env.observation_space)
    print("Action space:", train_env.action_space)
    print("Dynamics parameters:", train_env.get_parameters())

    if args.algorithm == "SAC":
        model = SAC("MlpPolicy", train_env, verbose=1, device=device)
    elif args.algorithm == "PPO":
        model = PPO("MlpPolicy", train_env, verbose=1, device=device)

    model.learn(total_timesteps=args.timesteps, log_interval=4)
    model.save(args.save_path)
    print(f"Modello salvato in {args.save_path}")


if __name__ == "__main__":
    main()