from agent import GMCAgent
from gym.envs.box2d import LunarLander

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TRAINED_WEIGHTS = 'trained_agent_weights.npy'


def main():
    print('Creating environment...')
    env = LunarLander()
    print('Done.')


    print('Loading agent...')
    agent = GMCAgent()
    agent.set_weights(np.load(TRAINED_WEIGHTS))

    print('Testing agent...')
    trained_scores = [agent.land(env, verbose=False, render=False) for _ in range(100)]
    print('Done.')


    print('Creating figures...')
    plt.figure()
    plt.hist([s[0] for s in trained_scores], bins=20)
    plt.ylabel('Count')
    plt.grid()
    plt.text(100, 20, f'Mean total reward: {np.mean([s[0] for s in trained_scores])}')
    _ = plt.xlabel('Total Rewards')
    plt.savefig('trained_figures/rewards_test.png')

    # plt.figure()
    # pd.Series(agent.history['rewards']).rolling(50, min_periods=1).mean().plot()
    # plt.xlabel('Episodes')
    # plt.ylabel('Total Reward')
    # plt.savefig('trained_figures/rewards_train.png')

    plt.figure()
    pd.Series([s[1] for s in trained_scores]).value_counts().plot(kind='barh')
    plt.ylabel('Final Reward')
    plt.grid()
    _ = plt.xlabel('Count')
    plt.savefig('trained_figures/success_count.png')

    plt.close('all')
    print('Done.')

if __name__ == '__main__':
    main()
