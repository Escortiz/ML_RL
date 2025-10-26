"""Test a random policy on the OpenAI Gym Hopper environment.

    
    TASK 1: Play around with this code to get familiar with the
            Hopper environment.

            For example:
                - What is the state space in the Hopper environment? Is it discrete or continuous?
                - What is the action space in the Hopper environment? Is it discrete or continuous?
                - What is the mass value of each link of the Hopper environment, in the source and target variants respectively?
                - what happens if you don't reset the environment even after the episode is over?
                - When exactly is the episode over?
                - What is an action here?
"""
import pdb
import os

import gym
from gym.wrappers import RecordVideo


from env.custom_hopper import *


def main():
	env_base = gym.make('CustomHopper-source-v0', render_mode="rgb_array")
	# env = gym.make('CustomHopper-target-v0', render_mode="rgb_array")
	video_folder = "./videos_hopper_gym021"
	os.makedirs(video_folder, exist_ok=True) # Crear carpeta si no existe
	
	env_base.metadata['video.frames_per_second'] = 250 # modify  	 for Record
	# Record every 100 episodes
	episode_trigger = lambda episode_id: episode_id % 100 == 0

	print('State space:', env_base.observation_space) # state-space
	print('Action space:', env_base.action_space) # action-space
	# Acceder al entorno sin el wrapper TimeLimit
	try:
		print('Dynamics parameters:', env_base.get_parameters()) # masses of each link of the Hopper
	except:
		print('Dynamics parameters: Not directly accessible due to wrappers')

	n_episodes = 500
	#render = True ## ------not necesary using RecordVideo ----
	env = RecordVideo(env_base, video_folder=video_folder, episode_trigger=episode_trigger)

	for episode in range(n_episodes):
		done = False
		state = env.reset()	# Reset environment to initial state
		step_count = 0
		while not done:  # Until the episode is over

			action = env.action_space.sample()	# Sample random action
		
			state, reward, done, info = env.step(action)	# Step the simulator to the next timestep
		
			#------- replaced by record video -----
			#if render:
			#	env.render(mode='rgb_array')

	env.close()

if __name__ == '__main__':
	main()