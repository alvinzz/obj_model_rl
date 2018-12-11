from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
import pickle
import matplotlib
import matplotlib.pyplot as plt
from railrl.images.camera import sawyer_init_camera
from gym import Env
import cv2

class PusherEnv(MujocoEnv, Serializable):
    FILE = 'pusher.xml'
    def __init__(self, choice=None, sparse = False , train = True):
        super(PusherEnv, self).__init__()
        self.frame_skip=2
        self.get_viewer()
        self.viewer_setup()

    def get_current_obs(self):
        #return np.concatenate([
        #    self.model.data.qpos.flat[:3],
        #    self.model.data.geom_xpos[-6:-1, :2].flat,
        #    self.model.data.qvel.flat,
        #]).reshape(-1)
        img = self.render(mode='rgb_array')
        img = cv2.resize(img, (0, 0), fx=64./(500), fy=64./(500), interpolation=cv2.INTER_AREA)
        return img.flatten()

    def relabeled_current_obs(self):
        img = self.get_current_obs().reshape(64, 64, 3)
        pusher_locs = np.where((img[:,:,0] == img[:,:,1]) & (img[:,:,0] > 0))
        img[pusher_locs] = [160,0,0]
        return img.flatten()

    def get_state(self):
        obj_pos = self.get_body_com("obj1")[:2]
        obj_pos = np.append(obj_pos, np.array([0])).flatten()
        obj_pos = np.reshape(obj_pos, (1,3))
        pusher = self.get_body_com("pusherObj")[:2]
        pusher = np.append(pusher, np.array([1])).flatten()
        pusher = np.reshape(pusher, (1,3))  
        assert np.shape(obj_pos) == (1,3,) and np.shape(pusher) == (1,3,)

        ret = np.concatenate((pusher, obj_pos))
        assert np.shape(ret) == (2,3)
        return ret

    def get_reward(self):
        return np.sum(np.abs(self.get_body_com('target')-self.get_body_com('obj1'))) + .3*np.sum(np.abs(self.get_body_com('pusherObj')-self.get_body_com('obj1')))

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        #img = env.render(mode='rgb_array')
        #img = cv2.resize(img, (0, 0), fx=64./(500), fy=64./(500), interpolation=cv2.INTER_AREA)
        observation = next_obs
        #observation = img.flatten()
        reward = self.get_reward()
        #reward = np.linalg.norm(self.get_body_com('obj1')-self.get_body_com('target'))
        done = False
        info = dict()

        return observation, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 1.2
        self.viewer.cam.azimuth = 90.0
        self.viewer.cam.elevation = -90.0

        self.viewer.cam.lookat[0] = 0.27
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0


class MultiPointmassEnv(MujocoEnv, Serializable):
    FILE = 'multipointmass.xml'
    def __init__(self, choice=None, sparse = False , train = True):
        super(MultiPointmassEnv, self).__init__()
        self.frame_skip=2
        self.get_viewer()
        self.viewer_setup()

    def get_current_obs(self):
        #return np.concatenate([
        #    self.model.data.qpos.flat[:3],
        #    self.model.data.geom_xpos[-6:-1, :2].flat,
        #    self.model.data.qvel.flat,
        #]).reshape(-1)
        img = self.render(mode='rgb_array')
        img = cv2.resize(img, (0, 0), fx=64./(500), fy=64./(500), interpolation=cv2.INTER_AREA)
        
        return img.flatten()

    def relabeled_current_obs(self):
        img = self.get_current_obs().reshape(64, 64, 3)
        background_locs = np.where((img[:,:,0] == img[:,:,1]) & (img[:,:,0] == img[:,:,2]) & (img[:,:,0] > 0))
        img[background_locs] = [0,0,0]
        yellow_locs = np.where((img[:,:,0] == img[:,:,1]) & (img[:,:,0] > 0))
        img[yellow_locs] = [255,0,0]
        green_locs = np.where((img[:,:,0] < img[:,:,1]))
        img[green_locs] = [0,255,0]
        blue_locs = np.where((img[:,:,2] > img[:,:,1]))
        img[blue_locs] = [0,0,255]
        return img.flatten()

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        #img = env.render(mode='rgb_array')
        #img = cv2.resize(img, (0, 0), fx=64./(500), fy=64./(500), interpolation=cv2.INTER_AREA)
        observation = next_obs
        #observation = img.flatten()
        reward = np.linalg.norm(self.get_body_com('obj1')[:2]-np.array([.1 ,-.1]))+np.linalg.norm(self.get_body_com('obj2')[:2]-np.array([-.1, .1]))

        done = False
        info = dict()

        return observation, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = .6
        self.viewer.cam.azimuth = 90.0
        self.viewer.cam.elevation = -90.0

        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0

if __name__ == "__main__":
    env = MultiPointmassEnv()
    #env = ImageMujocoEnv(env)
    hello = env.reset()
    print(hello.shape)
    # print("reset pusher: ", env.get_body_com("pusherObj"))
    # print("reset object: ", env.get_body_com("obj1"))
    for i in range(20):
        observation, reward, done, info = env.step(np.array([1,0]))
        print(observation.shape)
    env.reset()
    # print("reset pusher?: ", env.get_body_com("pusherObj"))
    # print("reset object?: ", env.get_body_com("obj1"))
    # env.render(mode='rgb_array')
