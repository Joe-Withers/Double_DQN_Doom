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
import cv2

def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
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

def run_agent(agent, game, frame_repeat):
    epsilon = 0.
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE
    total_reward = []
    for episode in range(episodes):
        i=0
        if episode > 150: epsilon = 0
        print("Episode #" + str(episode))
        reward_sum = 0
        game.new_episode()
        screen_buf = game.get_state().screen_buffer
        frame_manager = Frames(2,screen_buf)
        game.advance_action(1)
        screen_buf = game.get_state().screen_buffer
        # cv2.imwrite('./saved_sequence/'+str(episode)+'_'+str(i)+'.png',screen_buf)
        # plt.savefig('./saved_sequence/'+str(episode)+'_'+str(i)+'.png')
        stacked_frames = frame_manager.get_stacked_frames(screen_buf)
        state = np.expand_dims(stacked_frames,axis=0)
        done = game.is_episode_finished()
        while not done:
            i=i+1
            sleep(sleep_time)
            screen_buf = game.get_state().screen_buffer
            # cv2.imwrite('./saved_sequence/'+str(episode)+'_'+str(i)+'.png',screen_buf)
            # plt.imshow(screen_buf)
            # plt.savefig('./saved_sequence/'+str(episode)+'_'+str(i)+'.png')
            stacked_frames = frame_manager.get_stacked_frames(screen_buf)
            next_state = np.expand_dims(stacked_frames,axis=0)
            action = action_to_one_hot_bool(agent.select_epsilon_greedy_action(state,epsilon))
            for i in range(frame_repeat):
                sleep(sleep_time)
                reward = game.make_action(action, 1)
            done = game.is_episode_finished()
            state = next_state
            reward_sum += reward
            if done:
                print('episode:', episode, 'sum_of_rewards_for_episode:', reward_sum)
                total_reward.append(reward_sum)
                break
    return agent, total_reward

if __name__ == "__main__":
    version = 'DDQN'
    config_file = 'deadly_corridor'
    checkpoint_file = './results/dqn/agent1/cp-0001.ckpt'
    frame_repeat = 12
    episodes = 5
    config_path = 'D:/Joe/Anaconda3/envs/tensorflow_env/Lib/site-packages/vizdoom/scenarios/'+config_file+'.cfg'
    game = initialize_vizdoom(config_path)
    action_size = game.get_available_buttons_size()
    print('action_size',action_size)
    state_size = np.array(game.get_state().screen_buffer.shape)
    if version=='DQN':
        agent = DQN.DQN_Agent(state_size,action_size,checkpoint_file=checkpoint_file)
    if version=='DDQN':
        agent = Double_DQN.DQN_Agent(state_size,action_size,checkpoint_file=checkpoint_file)
    _, total_reward = run_agent(agent, game, frame_repeat)
    game.close()
