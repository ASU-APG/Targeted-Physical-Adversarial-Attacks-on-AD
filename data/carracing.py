import argparse
import math
from os import mkdir
from os.path import exists, join

import numpy as np
import torch

import gym
from data.utils import scenarios_seeds, scenarios_start_pos, sequence_length
from policy.agents.agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument('--scenario', type=str, default='straight', help='select driving scenario')
parser.add_argument('--same-track', action='store_true', help="Generates same track for every rollout if specified")
parser.add_argument('--rollouts', type=int, default=1, help="Number of rollouts")
parser.add_argument('--dir', type=str, help="Where to place rollouts")
parser.add_argument('--policy', type=str, choices=['pre', 'pre_noise', 'random_1', 'random_2'], default='pre',
                    help='Noise type used for action sampling')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--start-t', type=int, default=0, help="starting point of data collection")
args = parser.parse_args()

if args.scenario not in ['straight', 'left_turn', 'right_turn']:
    raise ValueError(
        f'Invalid scenario {args.scenario}. Please choose scenario either straight, left_turn or right_turn')

# initialize scenario specific settings
seed = scenarios_seeds[args.scenario]
start_t = scenarios_start_pos[args.scenario]

# cuda and seed setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Running on', device)
# np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

# pretrained policy parameters
RED_SIZE = 96
img_stack = 4
action_repeat = 8

# env and agent loading
env = gym.make("CarRacingAdv-v0", scenario=args.scenario)
if args.same_track:
    env.seed(seed)
agent = Agent(img_stack, device)
agent.load_param(device)


def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions


def rgb2gray(rgb, norm=True):
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray


def step(stack, action):
    img_list = []
    car_props_list = []
    rewards_list = []

    img_rgb = None
    done = False
    obj_loc = None
    for i in range(action_repeat):
        img_rgb, reward, done, _, adv, car_props, obj_loc = env.step(action)
        img_list.append(img_rgb)
        rewards_list.append(reward)
        car_props_list.append(car_props)
        if done:
            break
    img_gray = rgb2gray(img_rgb)
    stack.pop(0)
    stack.append(img_gray)
    assert len(stack) == img_stack
    return np.array(stack), rewards_list, done, car_props_list, obj_loc, img_list


def generate_data(rollouts, data_dir, policy):
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."

    i = 0
    track_seed = seed
    seq_len = sequence_length

    min_velx, min_vely = float('inf'), float('inf')
    max_velx, max_vely = float('-inf'), float('-inf')

    while i < rollouts:
        a_rollout = None
        if policy == 'random_1':
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif policy == 'random_2':
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)

        if args.same_track:
            env.seed(track_seed)
        s = env.reset()
        img_gray = rgb2gray(s)
        stack = [img_gray] * img_stack
        s = np.array(stack)

        s_rgb_rollouts = []
        r_rollouts = []
        car_props_rollouts = []
        actions_rollouts = []

        action = agent.select_action(s, device)
        if policy == 'pre_noise':
            action += np.random.normal(size=action.shape) * 0.15
            action = np.clip(action, 0, 1)
        elif policy in ['random_1', 'random_2']:
            action = a_rollout[0]

        s, r_list, done, car_props_list, \
        obj_loc, s_rgb_list = step(stack, action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
        # discard rollout if obj is not present in the track
        if len(obj_loc) == 0:
            if args.same_track:
                print('No object found from simulator. Retrying track with seed', track_seed + 1)
                track_seed += 1
            else:
                print('No object found from simulator. Retrying track')
            continue

        # waste few frames in the beginning
        for idx in range(start_t):
            action = agent.select_action(s, device)
            # if policy != 'pre':
            #     action += np.random.normal(size=action.shape) * 0.15
            #     action = np.clip(action, 0, 1)
            # elif policy in ['random_1', 'random_2']:
            #     action = a_rollout[idx]
            s, r_list, done, car_props_list, \
            obj_loc, s_rgb_list = step(stack, action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))

        actions_rollouts += [action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])] * action_repeat
        s_rgb_rollouts += s_rgb_list
        r_rollouts += r_list
        car_props_rollouts += car_props_list

        t = 1
        while True:
            action = agent.select_action(s, device)
            if policy != 'pre':
                action += np.random.normal(size=action.shape) * 0.15
                action = np.clip(action, 0, 1)
            elif policy in ['random_1', 'random_2']:
                action = a_rollout[t]

            s, r_list, done, car_props_list, \
            obj_loc, s_rgb_list = step(stack, action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            actions_rollouts += [action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])] * action_repeat
            s_rgb_rollouts += s_rgb_list
            r_rollouts += r_list
            car_props_rollouts += car_props_list
            obj_true_loc = [obj_loc]

            if args.render:
                env.render()

            if t == seq_len // action_repeat or done:
                print("> End of rollout {}, {} frames...".format(i, len(r_rollouts)))
                np.savez_compressed(join(data_dir, 'rollout_{}'.format(i)),
                                    observations=np.array(s_rgb_rollouts),
                                    rewards=np.array(r_rollouts),
                                    actions=np.array(actions_rollouts),
                                    car_props=np.array(car_props_rollouts),
                                    obj_loc=np.array(obj_true_loc))
                linvel = np.array([np.hstack(linvel) for linvel in np.array(car_props_rollouts)[:, 2]])
                min_velx = min(np.min(linvel[:, 0]), min_velx)
                min_vely = min(np.min(linvel[:, 1]), min_vely)
                max_velx = max(np.max(linvel[:, 0]), max_velx)
                max_vely = max(np.max(linvel[:, 1]), max_vely)
                break

            t += 1

        i += 1

    print('*' * 50)
    print(f'linVelx range: [{min_velx},{max_velx}]')
    print(f'linVely range: [{min_vely},{max_vely}]')


if __name__ == "__main__":
    directory = args.dir
    if not exists(directory):
        mkdir(directory)
    generate_data(args.rollouts, directory, args.policy)
