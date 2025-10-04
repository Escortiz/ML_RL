"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--n-episodes-save', default=1000, type=int, help='Number of episodes to save the model')
    

    return parser.parse_args()

args = parse_args()


def _unpack_reset(res):
    # Maneja gym / gymnasium: reset puede devolver obs o (obs, info)
    return res[0] if isinstance(res, tuple) and len(res) >= 1 else res


def _unpack_step(res):
    # Maneja step que devuelve 4-tupla (obs, reward, done, info) o 5-tupla (obs, reward, terminated, truncated, info)
    if isinstance(res, tuple):
        if len(res) == 4:
            return res
        if len(res) == 5:
            s, r, term, trunc, info = res
            done = term or trunc
            return s, r, done, info
    return res


def main():

    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    # Si tu entorno tiene get_parameters
    if hasattr(env, 'get_parameters'):
        print('Dynamics parameters:', env.get_parameters())


    """
        Training
    """
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

    for episode in range(args.n_episodes):
        done = False
        train_reward = 0.0
        state = _unpack_reset(env.reset())  # Reset the environment and observe the initial state

        while not done:  # Loop until the episode is over

            action, action_log_prob = agent.get_action(state)
            previous_state = state

            step_res = env.step(action.detach().cpu().numpy())
            state, reward, done, info = _unpack_step(step_res)

            agent.store_outcome(previous_state, state, action_log_prob, reward, done)

            train_reward += float(reward)


        #Al finalizar el episodio se actualiza la politica para REINFORCE
        #loss_value = agent.update_policy()

		# ----- REINFORCE WITH BASELINE 
        total_loss_val, actor_loss_val, critic_loss_val = agent.update_policy()
        
        if (episode+1)%args.print_every == 0:
            print('Training episode:', episode)
            print('Episode return:', train_reward)
            if 'total_loss_val' in locals(): 
                print(f'Total Loss: {total_loss_val:.4f}, Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}')

        if (episode+1)%args.n_episodes_save == 0:
            torch.save(agent.policy.state_dict(), f"model_episode_{episode+1}.mdl")
            print(f"Model saved at episode {episode+1} as model_episode_{episode+1}.mdl")
    torch.save(agent.policy.state_dict(), "model.mdl")


if __name__ == '__main__':
    main()