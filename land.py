from gym.envs.box2d import LunarLander

from agent import GMCAgent
import numpy as np

WEIGHTS = 'weights/trained_agent_weights_0.01_0.99.npy'

if __name__ == '__main__':
    agent = GMCAgent()
    agent.set_weights(np.load(WEIGHTS))
    env = LunarLander()
    agent.land(env)
