import vizdoom as vzd
from random import choice
from time import sleep
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from frame_manager import Frames
import Double_DQN
import DQN

def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.init()
    print("Doom initialized.")
    return game

def convert_done_to_bool(done):
    if done:
        return 0
    else:
        return 1

def action_to_one_hot_bool(action_int):
    action = [False]*action_size
    action[action_int] = True
    return action

def train_agent(agent, game, frame_repeat):
    epsilon = 0.6
    decay_rate = 0.99
    total_reward = []
    for episode in range(episodes):
        if episode > 300: epsilon = 0
        print("Episode #" + str(episode))
        reward_sum = 0
        game.new_episode()
        screen_buf = game.get_state().screen_buffer
        frame_manager = Frames(2,screen_buf)
        game.advance_action(1)
        screen_buf = game.get_state().screen_buffer
        stacked_frames = frame_manager.get_stacked_frames(screen_buf)
        state = np.expand_dims(stacked_frames,axis=0)
        done = game.is_episode_finished()
        while not done:
            screen_buf = game.get_state().screen_buffer
            stacked_frames = frame_manager.get_stacked_frames(screen_buf)
            next_state = np.expand_dims(stacked_frames,axis=0)
            action = action_to_one_hot_bool(agent.select_epsilon_greedy_action(state,epsilon))
            reward = game.make_action(action, frame_repeat)
            done = game.is_episode_finished()
            agent.add_replay((action,state,next_state,reward,convert_done_to_bool(done)))
            state = next_state
            reward_sum += reward
            if done:
                print('episode:', episode, 'sum_of_rewards_for_episode:', reward_sum)
                agent.learn_from_m_random_transitions_in_replay_buffer(m)
                epsilon = epsilon*decay_rate
                total_reward.append(reward_sum)
                break
    return agent, total_reward

if __name__ == "__main__":
    versions = ['DDQN']#,'DDQN']
    config_files = ['deadly_corridor']
    frame_repeat = 12
    #parameters
    batch_size=6
    learning_rate=0.001
    folder=str(learning_rate)
    replay_buffer_size=100000
    m = 256
    episodes = 50
    n_agents = 1
    for version in versions:
        for config_file in config_files:
            config_path = 'D:/Joe/Anaconda3/envs/tensorflow_env/Lib/site-packages/vizdoom/scenarios/'+config_file+'.cfg'
            all_total_rewards = []
            for n in range(n_agents):
                game = initialize_vizdoom(config_path)
                action_size = game.get_available_buttons_size()
                print('action_size',action_size)
                state_size = np.array(game.get_state().screen_buffer.shape)
                if version=='DQN':
                    agent = DQN.DQN_Agent(state_size,action_size, batch_size=batch_size, learning_rate=learning_rate, replay_buffer_size=replay_buffer_size, checkpoint_file='./agent'+str(n)+'/cp-9999.ckpt')
                if version=='DDQN':
                    agent = Double_DQN.DQN_Agent(state_size,action_size, batch_size=batch_size, learning_rate=learning_rate, replay_buffer_size=replay_buffer_size, checkpoint_file='./agent'+str(n)+'/cp-9999.ckpt')
                _, total_reward = train_agent(agent, game, frame_repeat)
                all_total_rewards.append(total_reward)
                game.close()

                all_total_rewards_to_save = np.array(all_total_rewards)
                if version=='DQN':
                    np.save('./all_total_rewards_'+config_file+'_'+version+'.npy', all_total_rewards_to_save)
                if version=='DDQN':
                    np.save('./all_total_rewards_'+config_file+'_'+version+'.npy', all_total_rewards_to_save)
