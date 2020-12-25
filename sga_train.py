from agent import GMCAgent, SemiGradientAgent, EpisodicSemiGradient, DummyTransformer
import matplotlib.pyplot as plt
import numpy as np
from utils import smoothen
import pandas as pd
from sklearn import pipeline, preprocessing
from datetime import datetime

from gym.envs.box2d import LunarLander

LR = 1e-8
EPS = 0.99
GAMMA = 0.999
MAX_STEPS = 1900
SUCCESS_COUNT = 5
SUCCESS_CRITERIA = 220


def main():
    env = LunarLander()

    # Add transformers for tile coding or extra features.
    transformer = pipeline.FeatureUnion(
        [
            # ('scaler', preprocessing.StandardScaler()),
            # ('square', preprocessing.FunctionTransformer(lambda x: x**2, validate=False)),
            # ('dummy', DummyTransformer()),
            # ('poly', preprocessing.PolynomialFeatures(2)),
            # ('cos', preprocessing.FunctionTransformer(np.cos, validate=False)),
            # ('inverter', preprocessing.FunctionTransformer(lambda x: 1. /(x + 1.), validate=False)),
            # ('quantile', preprocessing.KBinsDiscretizer(strategy='uniform', n_bins=20, encode='onehot')),
            # ('quantile-poly', pipeline.Pipeline([
            #     ('poly', preprocessing.PolynomialFeatures(2, interaction_only=True)),
            #     ('quantile', preprocessing.KBinsDiscretizer(strategy='quantile', n_bins=20, encode='onehot-dense', )),
            #     ])),
            # ('quantile', pipeline.Pipeline([
            #     # ('poly', preprocessing.PolynomialFeatures(2)),
            #     ('quantile', preprocessing.KBinsDiscretizer(strategy='uniform', n_bins=200, encode='ordinal', )),
            #     # ('ohe', preprocessing.OneHotEncoder(sparse=False, categories='auto'))
            #
            #     ])),
            # ('power', preprocessing.PowerTransformer()),

        ]
    )

    s = env.reset()

    # a = 1/(1000*s.T@s)
    print('Learning rate:', LR)

    agent = GMCAgent(
        lr=LR,
        init_epsilon=EPS,
        max_steps=MAX_STEPS,
        gamma=GAMMA,
        threshold=0.0,
        transformer=None,
        success_count=SUCCESS_COUNT,
        success_criteria=SUCCESS_CRITERIA
        )
    # agent = SemiGradientAgent(lr=LR, init_epsilon=EPS, max_steps=800, gamma=GAMMA, threshold=0.0,
    #                transformer=None, success_count=3, success_criteria=220)
    # agent = EpisodicSemiGradient(lr=a, init_epsilon=0.3, max_steps=800, gamma=0.9999, threshold=0.0,
    #                transformer=None, success_count=3, success_criteria=220)

    agent.fit(env, render_train=False, verbose=True, episodes=10000)

    agent.land(env, verbose=False)
    now = datetime.now().strftime("%Y-%m-%d")
    filename = f'weights/trained_agent_weights_{agent.alpha}_{agent.max_steps}_{now}'
    np.save(filename, agent.get_weights())
    print("Weights saved to", filename)
    # plt.plot(agent.history['rewards'])
    # plt.show()

if __name__ == '__main__':
    main()
