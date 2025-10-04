"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse
import os
import torch
import gym
from gym.wrappers import RecordVideo

from env.custom_hopper import *
from agent import Agent, Policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')

    # Nuevos argumentos para la grabación de video
    parser.add_argument('--record-video', default=True, action='store_true', help='Record videos of the episodes')
    parser.add_argument('--video-folder', default='./videos_test_agent', type=str, help='Folder to save recorded videos')
    parser.add_argument('--record-every', default=1, type=int, help='Record every N episodes (e.g., 1 to record all, 10 to record every 10th)')

    # Flag para activar entrenamiento (si quieres usar este script para entrenar)
    parser.add_argument('--train', default=False, action='store_true', help='Run training loop instead of evaluation')
    parser.add_argument('--save-every', default=50, type=int, help='Save model every N training episodes')

    return parser.parse_args()

args = parse_args()

def _unpack_reset(res):
    # Maneja compatibilidad con diferentes versiones de gym/gymnasium
    if isinstance(res, tuple) and len(res) >= 1:
        return res[0]
    return res

def _unpack_step(res):
    # step puede devolver 4-tupla (obs, reward, done, info) o 5-tupla (obs, reward, terminated, truncated, info)
    if isinstance(res, tuple):
        if len(res) == 4:
            return res  # (next_state, reward, done, info)
        if len(res) == 5:
            next_state, reward, terminated, truncated, info = res
            done = terminated or truncated
            return next_state, reward, done, info
    # Por si acaso
    return res

def main():
    # Crear entorno base
    env_base = gym.make('CustomHopper-source-v0')
    # env_base = gym.make('CustomHopper-target-v0') # Si quieres probar en el entorno target
    video_folder = "./videos_hopper_gym021"
    os.makedirs(video_folder, exist_ok=True) # Crear carpeta si no existe
    
    env_base.metadata['video.frames_per_second'] = 250 # modify  	 for Record
	# Record every 100 episodes
    episode_trigger = lambda episode_id: episode_id % 100 == 0

    # Configuración para la grabación de video
    if args.record_video:
        if not os.path.exists(args.video_folder):
            os.makedirs(args.video_folder)
        # Define quale episodi si registrano, si registra usando il arg record_every
        episode_trigger = lambda episode_id: (episode_id % args.record_every == 0)
        # Envolvemos el entorno base con RecordVideo
        # Si tu versión de gym necesita render_mode='rgb_array', ajústalo al crear el env, para poder grabar
        # Gym lo suele manejar automatico, pero si hay que especificar:
        # env_base = gym.make('CustomHopper-source-v0', render_mode='rgb_array')
        env = RecordVideo(env_base, video_folder=args.video_folder, episode_trigger=episode_trigger, name_prefix="test-agent-episode")
        print(f"Recording videos to {args.video_folder}, every {args.record_every} episode(s).")
    else:
        env = env_base # No grabar video, usar el entorno base directamente

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)

    # Acceder al entorno original si es wrapper, sino estamos grabando podemos obtener los parametros del env_base
    original_env = getattr(env, 'env', env)
    if hasattr(original_env, 'get_parameters'):
        print('Dynamics parameters:', original_env.get_parameters())

    # Obtener dimensiones de espacios
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)

    if args.model is None and not args.train:
        print("Error: No model path provided for evaluation. Use --model <path> or --train to train a model.")
        env.close()
        return

    if args.model is not None:
        try:
            policy.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)), strict=True)
            print(f"Model loaded from {args.model}")
        except FileNotFoundError:
            print(f"Warning: Model file not found at {args.model}; starting from random weights.")
        except Exception as e:
            print(f"Error loading model: {e}")
            env.close()
            return

    agent = Agent(policy, device=args.device)

    # TRAINING LOOP
    if args.train:
        for episode in range(args.episodes):
            done = False
            test_reward = 0.0
            state = _unpack_reset(env.reset())
            # Si tu gym devuelve (obs, info), _unpack_reset se encarga de ello
            while not done:
                action, _ = agent.get_action(state, evaluation=True)  
                step_res = env.step(action.detach().cpu().numpy())
                next_state, reward, done, info = _unpack_step(step_res)
                # Ajustar si step devuelve (obs, reward, terminated, truncated, info)
                agent.store_outcome(state, next_state, _, reward, done)
                state = next_state
                test_reward += float(reward)

                # Render solo si no se graba video
                if not args.record_video and args.render:
                    env_base.render()

            # Llamada para actualizar la política al finalizar el episodio
            update_ret = agent.update_policy()
            # update_policy puede devolver None o (total_loss, actor_loss, critic_loss)
            if isinstance(update_ret, tuple) and len(update_ret) == 3:
                loss, a_loss, c_loss = update_ret
                print(f"Train Episode {episode+1}/{args.episodes} | Return: {test_reward:.2f} | Loss: {loss:.4f}")
            else:
                print(f"Train Episode {episode+1}/{args.episodes} | Return: {test_reward:.2f}")

            # Guardar modelo periódicamente si se indicó path de guardado
            if args.model and ((episode + 1) % args.save_every == 0):
                out_dir = os.path.dirname(args.model)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                torch.save(policy.state_dict(), args.model)
                print(f"Saved model to {args.model}")

    # EVALUATION / TEST LOOP
    else:
        total_rewards = []
        for episode in range(args.episodes):
            done = False
            test_reward = 0.0
            state = _unpack_reset(env.reset())
            while not done:
                action, _ = agent.get_action(state, evaluation=True)  # determinista en evaluación
                step_res = env.step(action.detach().cpu().numpy())
                state, reward, done, info = _unpack_step(step_res)
                if not args.record_video and args.render:
                    env_base.render()
                test_reward += float(reward)

            print(f"Episode: {episode + 1}/{args.episodes} | Return: {test_reward:.2f}")
            total_rewards.append(test_reward)

        if total_rewards:
            print(f"\nAverage return over {len(total_rewards)} episodes: {sum(total_rewards)/len(total_rewards):.2f}")

    env.close()  # Es importante cerrar el entorno, especialmente si se está grabando video


if __name__ == '__main__':
    main()