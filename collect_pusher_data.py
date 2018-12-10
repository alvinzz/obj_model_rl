from envs.pusher import PusherEnv, MultiPointmassEnv
from umbrellas import gym
import numpy as np
import scipy as sp
import scipy.linalg
import tables
from matplotlib import transforms
import tqdm
import pickle
import numpy as np
env = PusherEnv()


N = 100
steps = 50
imsize = 64*64*3
action_size = 2


# filename = "test.pkl"
filename="pusher_states.npy"
filename_ = "pusher_actions.npy"
filename__ = "pusher_costs.npy"
# images = np.zeros((N, steps, imsize))
states = np.zeros((N, steps,2,3))
actions = np.zeros((N,steps,2))
costs = np.zeros((N,steps))





for i in tqdm.trange(N, desc='trajs'):
    obs = env.reset()

    for j in tqdm.trange(steps):
        state = env.get_state()
        # action= (np.random.normal(loc=[.5,0], scale=[1.3,.7]))
        action= (np.random.normal(loc=[0,0], scale=[1,1]))
        states[i,j] = state
        actions[i,j] = action 
        costs[i] = env.get_reward()
        obs, reward, _, info = env.step(
            action
        )





np.save(filename, states)
np.save(filename_, actions)
np.save(filename__, costs)

print('woohoo')
# with open(filename, 'wb') as fp:
    # pickle.dump(images, fp, protocol = pickle.HIGHEST_PROTOCOL)
