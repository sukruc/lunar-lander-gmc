import numpy as np


def get_random_states(env, iters=100, max_steps=800):
    random_states = [env.reset()]
    actions = np.arange(4)
    print(actions)
    for _ in range(iters):
        done = False
        i = 0
        env.reset()
        while not done and i < max_steps:
            s, r, done, _ = env.step(np.random.choice(actions))
            random_states.append(s)
            i += 1
    print('Number of collected states:', len(random_states))
    return np.array(random_states)


class DummyTransformer:
    def fit(self, x, y=None):
        return self

    def fit_transform(self, x, y=None):
        return x

    def transform(self, x, y=None):
        return x


class GMCAgent:
    def __init__(self, lr=0.01, gamma=0.99, init_epsilon=0.995, threshold=0.1,
                 max_steps=3000, transformer=None, decay_lr=True, success_criteria=150,
                 success_count=5):
        self.alpha = lr
        self.gamma = gamma
        self.epsilon = init_epsilon
        self.thresh = threshold
        self.w = None
        self.history = {'rewards': [], 'weights':[], 'gradients':[]}
        self.max_steps = max_steps
        if transformer is None:
            transformer = DummyTransformer()
        self.transformer = transformer
        self.transformer.fit(np.random.randn(1, 8))
        self.decay_lr = decay_lr
        self.success_criteria = success_criteria
        self.success_count = success_count

    def transform(self, s):
        return self.transformer.transform(s.reshape(-1,8))

    def Q(self, s, a):
        return self.transform(s.reshape(1,-1)) @ self.w[:, a == np.arange(4)]

    def V(self, s):
        return self.transform(s.reshape(1,-1)) @ self.w

    def policy(self, s):
        return (self.transform(s.reshape(1,-1)) @ self.w).argmax()

    def move(self, s):
        epsilon = max(self.epsilon, self.thresh)
        if np.random.random() < epsilon:
            return np.random.randint(self.w.shape[1])
        else:
            return self.policy(s)

    def _init_weights(self, env, initializer='zero'):
        if initializer == 'zero':
            self.w = np.zeros((self.transform(np.random.randn(1, env.observation_space.shape[0])).shape[1], env.action_space.n))
        if initializer == 'random':
            self.w = np.random.randn(self.transform(np.random.randn(1, env.observation_space.shape[0])).shape[1], env.action_space.n)


    def fit(self, env, episodes=1000, verbose=1, render_train=False):
        random_states = get_random_states(env)
        self.transformer.fit(random_states)
        del random_states

        if self.w is None:
            self._init_weights(env)
        for episode in range(episodes):
            if verbose:
                print('Episode:', episode)
            if self.history['rewards'] and np.mean(self.history['rewards'][-self.success_count:]) > self.success_criteria:
                print('Agent ready.')
                return
            self.fit_episode(env, verbose, render_train)
            self.epsilon = self.epsilon * 0.99

    def update_weights(self, S, A, R):
        # import pdb; pdb.set_trace()
        for i in range(len(S) - 1):
            gradient = (R[i]*(A[i] == np.arange(4)) + self.gamma * self.V(S[i + 1]) - self.V(S[i])).reshape(1,-1) * self.transform(S[i]).reshape(-1,1)
            # gradient = np.clip(gradient, -20, 20)
            self.w += self.alpha * gradient
            self.history['gradients'].append(np.abs(gradient).sum())
        gradient = (R[i+1]*(A[i+1] == np.arange(4)) - self.V(S[i+1])).reshape(1,-1) * self.transform(S[i+1]).reshape(-1,1)
        # gradient = np.clip(gradient, -20, 20)
        self.w += self.alpha * gradient
        self.history['gradients'].append(np.abs(gradient).sum())

    def fit_episode(self, env, verbose, render_train):
        S = [env.reset()]
        a = self.move(S[0])
        A = [a]
        R = []
        done = False
        i = 0
        while not done and i < self.max_steps:
            St, Rt, done, _ = env.step(a)
            S.append(St)
            R.append(Rt)
            a = self.move(St)
            if render_train:
                env.render()
            A.append(a)
            i += 1
        R.append(0)
        self.update_weights(S, A, R)
        sumr = sum(R)
        self.history['rewards'].append(sumr)
        if verbose:
            print('Total reward:', sumr)

    def get_weights(self):
        return self.w

    def set_weights(self, w):
        self.w = w

    def land(self, env, render=True, verbose=True):
        eps, thr = self.epsilon, self.thresh
        self.epsilon, self.thresh = 0., 0.
        done = False
        s = env.reset()
        S = []
        A = []
        if verbose:
            print('Initial state:')
            print(*s.round(2))
        i = 0
        rew = 0
        while not done and i < 2000:
            if verbose:
                print('State:')
                print(*s.round(2))
            a = self.move(s)
            S.append(s)
            A.append(a)
            if verbose:
                print('Action taken:', a)
            s, r, done, _ = env.step(a)
            if verbose:
                print('Reward:', round(r, 2))
            rew += r
            if render:
                env.render()
            i += 1
        self.epsilon, self.thresh= eps, thr
        return rew, r


class SemiGradientAgent(GMCAgent):
    def update_weights(self, s, a, r, st):
        # import pdb; pdb.set_trace()
        gradient = ((r)* (a == np.arange(4)   ) + self.gamma * self.V(st) - self.V(s)).reshape(1,-1) * self.transform(s).reshape(-1,1)
        # gradient = np.clip(gradient, -70, 70)
        # if len(self.history['rewards']) > 90:
        #     import pdb; pdb.set_trace()
        self.w += self.alpha * gradient
        self.history['gradients'].append(np.abs(gradient).sum())
    #
    # def _init_weights(self, env):
    #     self.w = np.zeros((9, 4))
    #
    # def transform(self, s):
    #     return self.transformer.transform(s.reshape(-1,9))

    def fit_episode(self, env, verbose, render_train):
        s = env.reset()
        done = False
        i = 0
        # s = np.array(s.tolist() + [i/500])
        total_rewards = 0
        while not done and i <= self.max_steps:
            a = self.policy(s)
            st, r, done, _ = env.step(a)
            # st = np.array(s
            # st = np.array(st.tolist() + [(i+1)/500])
            total_rewards += r
            # if i == self.max_steps:
            #     r = -200
            if render_train:
                env.render()
            self.update_weights(s, a, r, st)
            s = st
            i += 1
        self.history['rewards'].append(total_rewards)
        if verbose:
            print('Total rewards:', total_rewards)
            print('Average of last 20 rewards:', np.mean(self.history['rewards'][-20:]))

class EpisodicSemiGradient(GMCAgent):
    def fit_episode(self, env, verbose, render_train):
        s = env.reset()
        a = self.policy(s)
        done = False
        total_rewards = 0
        i = 0
        while not done and i <= self.max_steps:
            st, r, done, _ = env.step(a)
            if done:
                self.w = self.w + self.alpha * (R - self.Q(s, a)) * s.reshape(-1, 1)
            st = np.array(st.tolist() + [(i+1)/500])
            total_rewards += r
            # if i == self.max_steps:
            #     r = -200
            if render_train:
                env.render()
            self.update_weights(s, a, r, st)
            s = st
            i += 1
        self.history['rewards'].append(total_rewards)
        if verbose:
            print('Total rewards:', total_rewards)
            print('Average of last 20 rewards:', np.mean(self.history['rewards'][-50:]))
