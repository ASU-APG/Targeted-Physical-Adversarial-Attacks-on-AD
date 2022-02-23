import gym
import numpy as np


class Env():
    """
    Test environment wrapper for CarRacingAdv
    """

    def __init__(self, seed, img_stack, action_repeat):
        self.seed = seed
        self.img_stack = img_stack
        self.action_repeat = action_repeat
        self.env = gym.make('CarRacingAdv-v0')
        if seed is not None:
            self.env.seed(seed)
        self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        if self.seed is not None:
            self.env.seed(self.seed)
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack
        self.adv_stack = [np.zeros_like(img_gray)] * self.img_stack
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, die, _, adv, car_props, obj_poly = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        self.adv_stack.pop(0)
        self.adv_stack.append(self.rgb2gray(adv))
        assert len(self.stack) == self.img_stack
        self.stack = [img_gray] * self.img_stack
        return np.array(self.stack), total_reward, done, die, np.array(self.adv_stack), car_props, obj_poly

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
