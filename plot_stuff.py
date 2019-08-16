import matplotlib.pyplot as plt
import numpy as np

config_file = 'deadly_corridor'
versions = ['DQN','DDQN']
folders = ['0.001/dqn','0.001/ddqn']

for version, folder in zip(versions, folders):
    all_total_rewards = np.load('./'+folder+'/all_total_rewards_'+config_file+'_'+version+'.npy')

    b=1
    plt.plot(np.convolve(np.mean(all_total_rewards,axis=0),np.ones(b)/b,'valid'))


plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend(versions)
plt.savefig('temp1.png')
plt.show()

# for i in range(3):
#     for version, folder in zip(versions, folders):
#         all_total_rewards = np.load('./'+folder+'/all_total_rewards_'+config_file+'_'+version+'.npy')
#
#         b=20
#         plt.plot(np.convolve(all_total_rewards[i],np.ones(b)/b,'valid'))
#
#
#     plt.xlabel('Episode')
#     plt.ylabel('Average Reward')
#     plt.legend(versions)
#     plt.savefig('temp'+str(i)+'.png')
#     plt.show()
