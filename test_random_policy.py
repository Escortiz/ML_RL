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

import  gym
from gym.wrappers import RecordVideo


from env.custom_hopper import *


def main():
	env_base = gym.make('CustomHopper-source-v0', render_mode="rgb_array") #render mode rgb_array to capture frames
	# env = gym.make('CustomHopper-target-v0')
	video_folder = "./videos"
	# Record every 100 episodes
	episode_trigger = lambda episode_id: episode_id % 100 == 0

	print('State space:', env.observation_space) # state-space
	print('Action space:', env.action_space) # action-space
	print('Dynamics parameters:', env.get_parameters()) # masses of each link of the Hopper

	n_episodes = 500
	render = True
	env = RecordVideo(env_base, video_folder=video_folder, episode_trigger=episode_trigger)

	for episode in range(n_episodes):
		done = False
		state = env.reset()	# Reset environment to initial state

		while not done:  # Until the episode is over

			action = env.action_space.sample()	# Sample random action
		
			state, reward, done, info = env.step(action)	# Step the simulator to the next timestep

			if render:
				env.render()

	env.close()

if __name__ == '__main__':
	main()