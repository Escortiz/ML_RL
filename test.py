"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')

	# Nuevos argumentos para la grabación de video
    parser.add_argument('--record-video', default=False, action='store_true', help='Record videos of the episodes')
    parser.add_argument('--video-folder', default='./videos_test_agent', type=str, help='Folder to save recorded videos')
    parser.add_argument('--record-every', default=1, type=int, help='Record every N episodes (e.g., 1 to record all, 10 to record every 10th)')


    return parser.parse_args()

args = parse_args()


def main():

	env_base = gym.make('CustomHopper-source-v0')
    # env_base = gym.make('CustomHopper-target-v0') # Si quieres probar en el entorno target

    # Configuración para la grabación de video
	if args.record_video:
		if not os.path.exists(args.video_folder):
			os.makedirs(args.video_folder)
        # Define qué episodios grabar. Aquí grabamos según el argumento --record-every
        # Si args.record_every es 1, graba todos. Si es N, graba el 0, N, 2N, ...
		episode_trigger = lambda episode_id: (episode_id % args.record_every == 0)
        
        # Envolvemos el entorno base con RecordVideo
        # Nota: El render_mode del entorno base podría necesitar ser 'rgb_array' para RecordVideo.
        # Gym a menudo lo maneja automáticamente, pero si hay problemas, se puede especificar:
        # env_base = gym.make('CustomHopper-source-v0', render_mode='rgb_array')
		env = RecordVideo(env_base, video_folder=args.video_folder, episode_trigger=episode_trigger, name_prefix="test-agent-episode")
		print(f"Recording videos to {args.video_folder}, every {args.record_every} episode(s).")
	else:
		env = env_base # Usamos el entorno base si no se graba video

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
    
    # Si no estamos grabando video, podemos obtener los parámetros del env_base
    # Si estamos grabando, env es un wrapper, así que accedemos a env.env para el entorno original
	original_env = env if not args.record_video else env.env 
	print('Dynamics parameters:', original_env.get_parameters())

	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
    
	if args.model is None:
		print("Error: No model path provided. Please specify a model using --model <path_to_model.mdl>")
		env.close()
		return
        
	try:
		policy.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)), strict=True)
		print(f"Model loaded from {args.model}")
	except FileNotFoundError:
		print(f"Error: Model file not found at {args.model}")
		env.close()
		return
	except Exception as e:
		print(f"Error loading model: {e}")
		env.close()
		return


	agent = Agent(policy, device=args.device)

	total_rewards = []

	for episode in range(args.episodes):
		done = False
		test_reward = 0
		state = env.reset()

		while not done:
			action, _ = agent.get_action(state, evaluation=True)
			state, reward, done, info = env.step(action.detach().cpu().numpy())

			# Si --record-video NO está activo Y --render SÍ está activo, entonces renderizamos.
			# RecordVideo maneja su propio renderizado para la grabación.
			if not args.record_video and args.render:
				env_base.render() # Usamos env_base para renderizar en pantalla si no se está grabando

			test_reward += reward

		print(f"Episode: {episode + 1}/{args.episodes} | Return: {test_reward:.2f}")
		total_rewards.append(test_reward)
    
	if total_rewards:
		print(f"\nAverage return over {args.episodes} episodes: {sum(total_rewards)/len(total_rewards):.2f}")

	env.close() # Es importante cerrar el entorno, especialmente si se está grabando video


if __name__ == '__main__':
	main()