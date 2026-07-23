from asteroid import AsteroidStatic
import numpy as np
import matplotlib.pyplot as plt
import itertools

ACTIONS = [(-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0), (0.0, -1.0), (0.0, 0.0), (0.0, 1.0), (1.0, -1.0), (1.0, 0.0), (1.0, 1.0)]

NUM_ACTIONS =9
def get_action(i): return ACTIONS[i]


class CallCounter:
    def __init__(self, env):
        self.n = 0
        self._orig = env._calc_next_state
        env._calc_next_state = self._wrap
    def _wrap(self, a):
        self.n += 1
        return self._orig(a)


if __name__ == '__main__':
    

    env = AsteroidStatic(seed=0)
    obs = env.reset()
    counter = CallCounter(env)
   

    obs, reward, done, info = env.step([0.0, 0.0])
    total_reward = reward
    while not done:
        action = np.random.randint(0, 8)
        obs, reward, done, info = env.step(get_action(action))
        total_reward += reward
        if done: 
            print(total_reward)
            print(counter.n)
            print(info['reached_goal'])
            break
       