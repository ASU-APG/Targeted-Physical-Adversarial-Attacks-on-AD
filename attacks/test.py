import argparse
import math
from os import mkdir
from os.path import exists

import kornia
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from agents.agent import Agent
from env.env_adv import Env
from matplotlib.patches import Rectangle

from gym.envs.box2d.car_racing_adv import WINDOW_W, WINDOW_H, STATE_W, STATE_H

parser = argparse.ArgumentParser(description='Test different scenarios of physical adversarial attacks on CarRacing-v0')
parser.add_argument('--scenario', type=str, default='straight', help='select driving scenario')
parser.add_argument('--action-repeat', type=int, default=1, metavar='N', help='repeat action in N frames (default: 1)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--unroll-length', type=int, default=25, help='Number of timesteps to attack')
parser.add_argument('--adv-bound', type=float, default=0.9, help='Adversarial bound for perturbation')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--save', action='store_true', help='save the results if specified')
args = parser.parse_args()

# cuda and seeds
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# common variables
T = args.T
eps = args.eps
d_s_size_x, d_s_size_y = None, None
rand_d_s, d_s, target = None, None, None
start_pos = None
# figures save directory
save_dir = 'animations/'

# set environment specific variables
if args.scenario == 1:
    # straight track corresponding to warp_2 file. Works with seed 2
    d_s_size_x, d_s_size_y = 25, 30
    obj_true_loc = [(145, 90), (130, 90), (130, 72), (145, 72)]

    # # small object
    # d_s_size_x, d_s_size_y = 11, 20
    # obj_true_loc = [(137, 82), (130, 82), (130, 72), (137, 72)]
    # init random perturbation
    rand_d_s = np.random.uniform(low=-eps, high=eps, size=(1, 1, d_s_size_x, d_s_size_y)).astype(np.float32)
    # load optimized perturbation
    d_s = np.load(f'adv_attacks/perturbations/throwaway_phys_v5_adam_alt_static_warp_{T}_eps_{eps}.npz')['arr_0']
    print(np.min(d_s), np.max(d_s))
    target = np.load('adv_attacks/target_samples/target_2.npy')
    # position from which start state is chosen
    start_pos = 17 * 8 - 1
    save_dir += 'scenario_new_1'

    # To check the robustness with respect to different box location
    # obj_true_loc = [(x + 8, y) for (x, y) in obj_true_loc]
    # print(obj_true_loc)

    # # straight track old data. Not using now but anyway keeping
    # T = 56
    # d_s_size_x, d_s_size_y = 25, 30
    # obj_true_loc = [(195.01408570902476, -24.43932843113483), (213.74506774747402, -24.43932843113483),
    #                 (213.74506774747402, -9.996894916319441), (195.01408570902476, -9.996894916319441)]
    # obj_true_loc = [(194.01408570902476, -24.43932843113483), (215.74506774747402, -24.43932843113483),
    #                 (215.74506774747402, -7.996894916319441), (194.01408570902476, -7.996894916319441)]

    # # straight track new 2 corresponding to warp_3 file. Failed attack when object is on left and target is right
    # T = 25
    # d_s_size_x, d_s_size_y = 25, 30
    # obj_true_loc = [(127, 56), (112, 56), (112, 38), (127, 38)]
    # # load random perturbation
    # # d_s = np.random.uniform(low=-eps, high=eps, size=(1, 1, d_s_size_x, d_s_size_y)).astype(np.float32)
    # # load optimized perturbation
    # d_s = np.load(f'adv_attacks/perturbations/throwaway_phys_v5_adam_alt_static_warp_3_{T}.npz')['arr_0']
    # # plt.imshow(d_s.squeeze(), cmap='gray')
    # # plt.show()
    # target = np.load('adv_attacks/target_samples/target_2.npy')

elif args.scenario == 2:
    # # left track corresponding to left_track_static_warp file. Works with seed 2
    # d_s_size_x, d_s_size_y = 34, 39
    # obj_true_loc = [(252.1328294371665, -5.887972113311335), (232.45142478132817, -5.887972113311335),
    #                 (232.45142478132817, 15.734467537077343), (252.1328294371665, 15.734467537077343)]

    # left track corresponding to left_track_static_warp_2 file. Works with seed 2
    d_s_size_x, d_s_size_y = 25, 30
    obj_true_loc = [(252, 5), (236, 5), (236, 20), (252, 20)]
    # init random perturbation
    rand_d_s = np.random.uniform(low=-eps, high=eps, size=(1, 1, d_s_size_x, d_s_size_y)).astype(np.float32)
    # load optimized perturbation
    d_s = np.load(f'adv_attacks/perturbations/throwaway_phys_v5_adam_alt_left_track_static_warp_2_{T}.npz')['arr_0']
    target = np.load('adv_attacks/target_samples/target_4.npy')
    # position from which start state is chosen
    start_pos = 8 * 8
    save_dir += 'scenario_new_2'

    # # To check the robustness with respect to different box location
    # obj_true_loc = [(x, y - 8) for (x, y) in obj_true_loc]
    # print(obj_true_loc)

    # # left track 2. Not sure why this is but keeping it anyway
    # d_s_size_x, d_s_size_y = 34, 39
    # obj_true_loc = [(195.1328294371665, -7.887972113311335), (215.45142478132817, -7.887972113311335),
    #                 (215.45142478132817, 9.734467537077343), (195.1328294371665, 9.734467537077343)]
elif args.scenario == 3:
    """
    For scenario 3 it is required to manually change the starting position of car to 178 in car_racing_adv.py file
    """
    # right track corresponding to right_track_static_warp file. works with seed 3
    d_s_size_x, d_s_size_y = 25, 30
    obj_true_loc = [(14.171737744295465, -28.644126878945876), (32.663034952274376, -28.644126878945876),
                    (32.663034952274376, -13.58142654032553), (14.171737744295465, -13.58142654032553)]
    # init random perturbation
    rand_d_s = np.random.uniform(low=-eps, high=eps, size=(1, 1, d_s_size_x, d_s_size_y)).astype(np.float32)
    # load optimized perturbation
    d_s = np.load(f'adv_attacks/perturbations/throwaway_phys_v5_adam_alt_right_track_static_warp_{T}.npz')['arr_0']
    target = np.load('adv_attacks/target_samples/target_9.npy')
    # position from which start state is chosen
    start_pos = 8 * 8
    save_dir += 'scenario_new_3'

    # # To check the robustness with respect to different box location
    # obj_true_loc = [(x + 2, y + 2) for (x, y) in obj_true_loc]
    # # obj_true_loc = [(x - 4, y - 4) for (x, y) in obj_true_loc]
    # print(obj_true_loc)

if not exists(save_dir):
    mkdir(save_dir)


def get_obj_params(car_props, obj_true_loc):
    """
    To get object parameters from car dynamics and given true object location
    :param car_props:
    :param obj_true_loc:
    :return:
    """
    zoom = 16.200000000000003
    x = car_props[0].unsqueeze(0).unsqueeze(0)
    y = car_props[1].unsqueeze(0).unsqueeze(0)
    angle = car_props[2]
    linVelx = car_props[3]
    linVely = car_props[4]

    angle = -angle
    if torch.norm(torch.tensor([linVelx, linVely])) > 0.5:
        angle = torch.atan2(linVelx, linVely)
    obj_state_params = None
    for idx in range(len(obj_true_loc)):
        tmp_x = obj_true_loc[idx][0] - x
        tmp_y = obj_true_loc[idx][1] - y
        obj_x = zoom * (tmp_x * torch.cos(angle) - tmp_y * torch.sin(angle)) / WINDOW_W * STATE_W + STATE_W / 2
        obj_y = zoom * (tmp_x * torch.sin(angle) + tmp_y * torch.cos(angle)) / WINDOW_H * STATE_H + STATE_H / 4
        if idx == 0:
            obj_state_params = torch.cat([obj_x, obj_y], dim=1)
        else:
            obj_state_params = torch.cat([obj_state_params, torch.cat([obj_x, obj_y], dim=1)])

    obj_state_params -= torch.tensor([0., 96.], device=device)
    obj_state_params[:, 1] = -obj_state_params[:, 1]
    return obj_state_params


def run_env_no_object(env, agent):
    t0, tT = None, None
    actions = []
    state = env.reset()
    reward_subset = 0
    agent_states = []

    trajectories = []

    i = 0
    score = 0
    T = args.T
    for t in range(1000):
        if t >= start_pos:
            if i == T:
                tT = state
                break
            if i < T:
                if i == 0:
                    t0 = state
                i += 1
            trajectories.append(state)
        if args.render:
            env.render()
        action = agent.select_action(state, device)
        action = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
        state_, reward, done, die, _, car_props, obj_poly = env.step(action)
        if 1 <= i <= T:
            actions.append(action)
            reward_subset += reward
            (x, y), angle, (linVelx, linVely), omega = car_props[0], car_props[1], car_props[2], car_props[3]
            agent_states.append([x, y, angle, linVelx, linVely])
        score += reward
        state = state_
        if done or die:
            break

    # print(agent_states)
    np.savez_compressed(f'{save_dir}/clean_{args.T}_eps_{eps}', np.array(trajectories))
    print('Score: {:.2f}\t'.format(score))
    return t0, tT, actions, reward_subset, agent_states


# def run_env(env, agent, mod_type='random'):
#     t0, tT = None, None
#     perturb = None
#     actions = []
#     reward_subset = 0
#     agent_states = []
#
#     if mod_type == 'random':
#         perturb = rand_d_s
#     elif mod_type == 'attack':
#         perturb = d_s
#
#     # v flip
#     # perturb = np.expand_dims(np.expand_dims(np.flip(perturb.squeeze(), 0), 0), 0)
#     # h flip
#     # perturb = np.expand_dims(np.expand_dims(np.flip(perturb.squeeze(), 1), 0), 0)
#     # plt.imshow(perturb.squeeze(), cmap='gray')
#     # plt.show()
#     #
#     # plt.imshow(np.flip(perturb.squeeze(), 1), cmap='gray')
#     # plt.show()
#
#     state = env.reset()
#     mask_rep_np = np.zeros_like(state)
#     mask_w_rep_np = np.zeros_like(state)
#
#     i = 0
#     score = 0
#     for t in range(1000):
#         if t >= start_pos:
#             if i == T:
#                 tT = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
#                 break
#             if i < T:
#                 if i == 0:
#                     # warped_obs = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
#                     t0 = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
#                 state = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
#                 i += 1
#                 state.clip(-1, 0.9921875)
#                 # import matplotlib.pyplot as plt
#                 # plt.imshow(state[0], cmap='gray')
#                 # plt.pause(0.000000000001)
#                 # # plt.show()
#                 # if i == 1 or i == T:
#                 #     plt.show()
#                 # # plt.show()
#                 # plt.clf()
#         if args.render:
#             env.render()
#         action = agent.select_action(state, device)
#         action = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
#         state_, reward, done, die, _, car_props, obj_poly = env.step(action)
#         if 1 <= i <= T:
#             actions.append(action)
#             reward_subset += reward
#             (x, y), angle, (linVelx, linVely), omega = car_props[0], car_props[1], car_props[2], car_props[3]
#             agent_states.append([x, y, angle, linVelx, linVely])
#             # print(reward)
#
#         (x, y), angle, (linVelx, linVely) = car_props[0], car_props[1], car_props[2]
#         obj_state_params = get_obj_params(torch.tensor([x, y, angle, linVelx, linVely]), obj_true_loc)
#
#         points_src = torch.tensor([[[0., 0.], [d_s_size_y, 0.], [d_s_size_y, d_s_size_x], [0., d_s_size_x]]])
#         M: torch.tensor = kornia.get_perspective_transform(points_src, obj_state_params.unsqueeze(0))
#         mask: torch.tensor = kornia.warp_affine(torch.tensor(perturb.copy()), M[:, :2, :], dsize=(96, 96))
#         mask_rep = torch.repeat_interleave(mask, 4, dim=0).reshape(4, 96, 96)
#         mask_rep_np = mask_rep.detach().numpy()
#
#         mask_w: torch.tensor = kornia.warp_affine(torch.ones(*perturb.shape), M[:, :2, :], dsize=(96, 96))
#         # if i == 7:
#         #     plt.imshow(mask_w.detach().squeeze().numpy(), cmap='gray')
#         #     plt.axis('off')
#         #     plt.show()
#         mask_w_rep = torch.repeat_interleave(mask_w, 4, dim=0).reshape(4, 96, 96)
#         mask_w_rep_np = mask_w_rep.detach().numpy()
#
#         score += reward
#         state = state_
#         if done or die:
#             break
#
#     print(agent_states)
#     print('Score: {:.2f}\t'.format(score))
#     return t0, tT, actions, reward_subset, agent_states


def run_env(env, agent, mod_type='random'):
    t0, tT = None, None
    perturb = None
    actions = []
    reward_subset = 0
    agent_states = []
    state_rgb = None
    trajectories = []

    if mod_type == 'random':
        perturb = rand_d_s
    elif mod_type == 'attack':
        perturb = d_s

    # v flip
    # perturb = np.expand_dims(np.expand_dims(np.flip(perturb.squeeze(), 0), 0), 0)
    # h flip
    # perturb = np.expand_dims(np.expand_dims(np.flip(perturb.squeeze(), 1), 0), 0)
    # plt.imshow(perturb.squeeze(), cmap='gray')
    # plt.show()
    #
    # plt.imshow(np.flip(perturb.squeeze(), 1), cmap='gray')
    # plt.show()

    state = env.reset()
    mask_rep_np = np.zeros_like(state)
    mask_w_rep_np = np.zeros_like(state)

    i = 0
    score = 0
    # import time
    # time.sleep(10)
    T = args.T
    for t in range(1000):
        if t >= start_pos:
            if i == T:
                tT = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
                # tT_rgb = state_rgb / 255. * (1 - np.moveaxis(mask_w_rep_np[1:], 0, -1)) + np.moveaxis(mask_w_rep_np[1:], 0, -1) * np.moveaxis(mask_rep_np[1:], 0, -1)
                break
            if i < T:
                if i == 0:
                    # warped_obs = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
                    t0 = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
                    # t0_rgb = state_rgb / 255. * (1 - np.moveaxis(mask_w_rep_np[1:], 0, -1)) + np.moveaxis(mask_w_rep_np[1:], 0, -1) * np.moveaxis(mask_rep_np[1:], 0, -1)
                state = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
                i += 1
                state.clip(-1, 0.9921875)
                # import matplotlib.pyplot as plt
                # plt.imshow(state[0], cmap='gray')
                # plt.pause(0.000000000001)
                # # plt.show()
                # if i == T:
                #     plt.show()
                # # plt.show()
                # plt.clf()
            trajectories.append(state)
        if args.render:
            env.render()
        action = agent.select_action(state, device)
        action = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
        state_, reward, done, die, _, car_props, obj_poly = env.step(action)
        if 1 <= i <= T:
            actions.append(action)
            reward_subset += reward
            (x, y), angle, (linVelx, linVely), omega = car_props[0], car_props[1], car_props[2], car_props[3]
            agent_states.append([x, y, angle, linVelx, linVely])
            # print(reward)

        (x, y), angle, (linVelx, linVely) = car_props[0], car_props[1], car_props[2]
        obj_state_params = get_obj_params(torch.tensor([x, y, angle, linVelx, linVely]), obj_true_loc)

        points_src = torch.tensor([[[0., 0.], [d_s_size_y, 0.], [d_s_size_y, d_s_size_x], [0., d_s_size_x]]])
        M: torch.tensor = kornia.get_perspective_transform(points_src, obj_state_params.unsqueeze(0))
        mask: torch.tensor = kornia.warp_affine(torch.tensor(perturb.copy()), M[:, :2, :], dsize=(96, 96))
        mask_rep = torch.repeat_interleave(mask, 4, dim=0).reshape(4, 96, 96)
        mask_rep_np = mask_rep.detach().numpy()

        mask_w: torch.tensor = kornia.warp_affine(torch.ones(*perturb.shape), M[:, :2, :], dsize=(96, 96))
        # if i == 7:
        #     plt.imshow(mask_w.detach().squeeze().numpy(), cmap='gray')
        #     plt.axis('off')
        #     plt.show()
        mask_w_rep = torch.repeat_interleave(mask_w, 4, dim=0).reshape(4, 96, 96)
        mask_w_rep_np = mask_w_rep.detach().numpy()

        score += reward
        state = state_
        if done or die:
            break

    # print(agent_states)
    np.savez_compressed(f'{save_dir}/adv_{args.T}_eps_{eps}', np.array(trajectories))
    print('Score: {:.2f}\t'.format(score))
    return t0, tT, actions, reward_subset, agent_states


def run_env_random(env, agent):
    perturb = None
    agent_states_multi = []

    # v flip
    # perturb = np.expand_dims(np.expand_dims(np.flip(perturb.squeeze(), 0), 0), 0)
    # h flip
    # perturb = np.expand_dims(np.expand_dims(np.flip(perturb.squeeze(), 1), 0), 0)
    # plt.imshow(perturb.squeeze(), cmap='gray')
    # plt.show()
    #
    # plt.imshow(np.flip(perturb.squeeze(), 1), cmap='gray')
    # plt.show()

    for i_ep in range(10):
        perturb = np.random.normal(0, 0.3, size=(1, 1, d_s_size_x, d_s_size_y)).astype(np.float32)
        state = env.reset()
        mask_rep_np = np.zeros_like(state)
        mask_w_rep_np = np.zeros_like(state)
        agent_states = []
        i = 0
        score = 0
        for t in range(1000):
            if t >= start_pos:
                if i == T:
                    tT = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
                    break
                if i < T:
                    if i == 0:
                        # warped_obs = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
                        t0 = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
                    state = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
                    i += 1
                    state.clip(-1, 0.9921875)
                    # import matplotlib.pyplot as plt
                    # plt.imshow(state[0], cmap='gray')
                    # plt.pause(0.000000000001)
                    # # plt.show()
                    # if i == 1 or i == T:
                    #     plt.show()
                    # # plt.show()
                    # plt.clf()
            if args.render:
                env.render()
            action = agent.select_action(state, device)
            action = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
            state_, reward, done, die, _, car_props, obj_poly = env.step(action)
            if 1 <= i <= T:
                (x, y), angle, (linVelx, linVely), omega = car_props[0], car_props[1], car_props[2], car_props[3]
                agent_states.append([x, y, angle, linVelx, linVely])
                # print(reward)

            (x, y), angle, (linVelx, linVely) = car_props[0], car_props[1], car_props[2]
            obj_state_params = get_obj_params(torch.tensor([x, y, angle, linVelx, linVely]), obj_true_loc)

            points_src = torch.tensor([[[0., 0.], [d_s_size_y, 0.], [d_s_size_y, d_s_size_x], [0., d_s_size_x]]])
            M: torch.tensor = kornia.get_perspective_transform(points_src, obj_state_params.unsqueeze(0))
            mask: torch.tensor = kornia.warp_affine(torch.tensor(perturb.copy()), M[:, :2, :], dsize=(96, 96))
            mask_rep = torch.repeat_interleave(mask, 4, dim=0).reshape(4, 96, 96)
            mask_rep_np = mask_rep.detach().numpy()

            mask_w: torch.tensor = kornia.warp_affine(torch.ones(*perturb.shape), M[:, :2, :], dsize=(96, 96))
            # if i == 7:
            #     plt.imshow(mask_w.detach().squeeze().numpy(), cmap='gray')
            #     plt.axis('off')
            #     plt.show()
            mask_w_rep = torch.repeat_interleave(mask_w, 4, dim=0).reshape(4, 96, 96)
            mask_w_rep_np = mask_w_rep.detach().numpy()

            score += reward
            state = state_
            if done or die:
                break

        agent_states_multi.append(np.array(agent_states))
        print('Score: {:.2f}\t'.format(score))

    agent_states_multi_mean = np.mean(agent_states_multi, axis=0)
    agent_states_multi_std = np.std(agent_states_multi, axis=0)
    print(agent_states_multi_mean.shape, agent_states_multi_std.shape)
    print(agent_states_multi_std)
    return agent_states_multi_mean, agent_states_multi_std


def plot_images(fig_list, fig_names, save):
    if save:
        matplotlib.use('Agg')
    for i in range(len(fig_list)):
        figure, fig_name = fig_list[i], fig_names[i]
        plt.imshow(figure[3], cmap='gray')
        # to highlight position of car
        plt.gca().add_patch(
            Rectangle((STATE_W / 2 - 5, 96 - STATE_H / 4 - 7), 9, 13, linewidth=2, edgecolor='r', facecolor='none'))
        if not save:
            plt.title(fig_name)
        plt.axis('off')
        plt.show()
        if save:
            plt.savefig(f'{save_dir}/{fig_name}.png', bbox_inches='tight')
        plt.close()


def plot_actions(actions_list, names, save):
    if save:
        matplotlib.use('Agg')
    colors = ['g-', 'b-', 'rx-']
    for idx in range(len(actions_list)):
        actions = actions_list[idx]
        steer = [action[0] for action in actions]
        plt.plot(np.arange(T), np.array(steer), colors[idx])
        plt.xlabel('Time steps', fontsize=15)
        plt.ylabel('Steering angle', fontsize=15)
        plt.ylim(-1, 1)  # -1 for left and 1 for right
    plt.legend(names, loc='upper left', prop={'size': 15})
    plt.grid()
    plt.show()
    if save:
        plt.savefig(f'{save_dir}/steer.png', bbox_inches='tight')
    plt.close()


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    print(origin, point, angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def plot_trajectories(a_state_list, names, save):
    if save:
        matplotlib.use('Agg')
    colors = ['g--', 'b--', 'r--']
    for idx in range(len(a_state_list)):
        a_states = a_state_list[idx]
        positions = np.array([a_state[:2] for a_state in a_states])
        positions -= positions[0]
        # rot_points = [positions[0]]
        # for i, point in enumerate(positions[1:]):
        #     rot_points.append(rotate(point, positions[0], a_states[0][2]))
        # rot_points = np.array(rot_points)
        base = plt.gca().transData
        from matplotlib import transforms
        rot = transforms.Affine2D().rotate(-a_states[0][2])
        # plt.rcParams['axes.facecolor'] = 'green'
        plt.plot(positions[:, 0], positions[:, 1], colors[idx], transform=rot + base, lw=2)
        if idx == 0:
            plt.scatter(positions[:1, 0], positions[:1, 1], c='black', transform=rot + base)
        if idx == 2:
            plt.scatter(positions[-1:, 0], positions[-1:, 1], c=colors[idx].replace('--', ''), marker='*', s=150,
                        transform=rot + base)
        else:
            plt.scatter(positions[-1:, 0], positions[-1:, 1], c=colors[idx].replace('--', ''), transform=rot + base,
                        label='_nolegend_')
        plt.tick_params(axis=u'both', which=u'both', length=0)
        # plt.axis('off')
        plt.xlabel('X Position', fontsize=15)
        plt.ylabel('Y Position', fontsize=15)
        if args.scenario == 1:
            plt.xlim(-20, 15)
            # plt.ylim(-10, 50)
        elif args.scenario == 2:
            plt.xlim(-20, 15)
            # plt.ylim(-10, 50)
        elif args.scenario == 3:
            plt.xlim(-20, 15)
            # plt.ylim(-10, 50)
        # plt.ylim(-1, 1)  # -1 for left and 1 for right
    plt.legend(names + ['Start', 'Target'], loc='lower left', prop={'size': 15})
    plt.grid(alpha=0.5)
    plt.show()
    if save:
        plt.savefig(f'{save_dir}/trajectory.png', bbox_inches='tight')
    plt.close()


def plot_trajectories_rand(a_state_list, names, std, save):
    if save:
        matplotlib.use('Agg')
    colors = ['g--', 'b--', 'r--']
    for idx in range(len(a_state_list)):
        a_states = a_state_list[idx]
        positions = np.array([a_state[:2] for a_state in a_states])
        positions -= positions[0]
        base = plt.gca().transData
        from matplotlib import transforms
        rot = transforms.Affine2D().rotate(-a_states[0][2])
        # plt.rcParams['axes.facecolor'] = 'green'
        plt.plot(positions[:, 0], positions[:, 1], colors[idx], transform=rot + base, lw=4)
        # if idx == 1:
        #     plt.plot(positions[:, 0] - std[:, 0], positions[:, 1] - std[:, 1], colors[idx].replace('--', ':'), transform=rot + base, lw=1, label='_nolegend_')
        #     plt.plot(positions[:, 0] + std[:, 0], positions[:, 1] + std[:, 1], colors[idx].replace('--', ':'), transform=rot + base, lw=1, label='_nolegend_')
        if idx == 0:
            plt.scatter(positions[:1, 0], positions[:1, 1], c='black', s=200, transform=rot + base)
        if idx == 2:
            plt.scatter(positions[-1:, 0], positions[-1:, 1], c=colors[idx].replace('--', ''), marker='*', s=200,
                        transform=rot + base)
        else:
            plt.scatter(positions[-1:, 0], positions[-1:, 1], c=colors[idx].replace('--', ''), transform=rot + base,
                        label='_nolegend_')
        plt.tick_params(axis=u'both', which=u'both', length=0)
        # plt.axis('off')
        plt.xlabel('X Position', fontsize=20)
        plt.ylabel('Y Position', fontsize=20)
        if args.scenario == 1:
            plt.xlim(-26, 15)
            # plt.ylim(-10, 50)
        elif args.scenario == 2:
            plt.xlim(-26, 15)
            # plt.ylim(-10, 50)
        elif args.scenario == 3:
            plt.xlim(-26, 15)
            # plt.ylim(-10, 50)
        # plt.ylim(-1, 1)  # -1 for left and 1 for right
    plt.legend(names + ['Start', 'Target'], loc='lower left', prop={'size': 18})
    plt.grid(alpha=0.5)
    plt.show()
    if save:
        plt.savefig(f'{save_dir}/trajectory.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    agent = Agent(args.img_stack, device)
    agent.load_param(device)

    env = Env(args.seed, args.img_stack, args.action_repeat)

    # # show or save target state
    # plot_images([target], ['target'], args.save)

    # run policy on env without object
    s_0, s_t, a_no_obj, rew_no_obj, a_state_no_obj = run_env_no_object(env, agent)
    # plot_images([s_0, s_t], ['start_no_obj', 'end_no_obj'], args.save)

    # run policy on env in the presence of random noise
    # rand_s_0, rand_s_t, a_rand, rew_rand, a_states_rand = run_env(env, agent, mod_type='random')  # mode takes 'random', 'attack'
    # plot_images([rand_s_0, rand_s_t], ['start_random', 'end_random'], args.save)

    # run multiple random and get average
    # a_states_rand_mean, a_states_rand_std = run_env_random(env, agent)

    # run policy on env in the presence of attack
    attack_s_0, attack_s_t, a_attack, rew_attack, a_states_attack = run_env(env, agent, mod_type='attack')
    # plot_images([attack_s_0, attack_s_t], ['start_attack', 'end_attack'], args.save)

    # plot trajectories using agent states
    # plot_trajectories_rand([a_state_no_obj, a_states_rand_mean, a_states_attack], ['No Object', 'Random Noise', 'Attack'], a_states_rand_std, args.save)

    # # plot steering angle
    # plot_actions([a_no_obj, a_rand, a_attack], ['No Object', 'Random Noise', 'Attack'], args.save)

    # intro_path = 'animations'
    # # for intro
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # axes[0].imshow(attack_s_0)
    # axes[0].axis('off')
    # axes[1].imshow(attack_s_t)
    # axes[1].axis('off')
    # plt.show()
    # if args.save:
    #     matplotlib.use('Agg')
    #     fig.savefig(f'{intro_path}/intro_full.png', bbox_inches='tight')

    # # action error
    # print('Action MSE')
    # a_err_rand = np.linalg.norm((np.array(a_no_obj) - np.array(a_rand) ** 2) / T)
    # a_err_attack = np.linalg.norm((np.array(a_no_obj) - np.array(a_attack) ** 2) / T)
    # print(f'Scenario {args.scenario}, Action error random noise {a_err_rand}')
    # print(f'Scenario {args.scenario}, Action error attack {a_err_attack}')
    #
    # # reward reduction
    # print('Reward Change')
    # print(rew_no_obj, rew_rand, rew_attack)
    # reward_red_random = (rew_rand - rew_no_obj) * 100 / rew_no_obj
    # reward_red_attack = (rew_attack - rew_no_obj) * 100 / rew_no_obj
    # print(f'Scenario {args.scenario}, Reward reduction random noise {reward_red_random}')
    # print(f'Scenario {args.scenario}, Reward reduction attack {reward_red_attack}')

    # running_score = 0
    # state = env.reset()
    #
    # # # for diagram
    # # obs = None
    # # patch = None
    # # warped_obs = None
    # # next_obs = None
    #
    # # for scenarios
    # start_obs = None
    # random_t0 = None
    # attack_t0 = None
    # random_tT = None
    # attack_tT = None
    #
    # # # with random baseline
    # # print('Running random baseline')
    # # rand_score = 0
    # # i = 0
    # # mask_rep_np = np.zeros_like(state)
    # # mask_w_rep_np = np.zeros_like(state)
    # # for t in range(1000):
    # #     if t >= 8 * 8:
    # #         if i == T:
    # #             random_tT = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
    # #             break
    # #         if i < T:
    # #             if i == 0:
    # #                 start_obs = state
    # #                 random_t0 = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
    # #             state = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
    # #             i += 1
    # #             state.clip(-1, 0.9921875)
    # #     if args.render:
    # #         env.render()
    # #     action = agent.select_action(state, device)
    # #     print(t, action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
    # #     state_, reward, done, die, _, car_props, obj_poly = env.step(
    # #         action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
    # #
    # #     (x, y), angle, (linVelx, linVely) = car_props[0], car_props[1], car_props[2]
    # #     obj_state_params = get_obj_params(torch.tensor([x, y, angle, linVelx, linVely]), obj_true_loc)
    # #
    # #     points_src = torch.tensor([[[0., 0.], [d_s_size_y, 0.], [d_s_size_y, d_s_size_x], [0., d_s_size_x]]])
    # #     M: torch.tensor = kornia.get_perspective_transform(points_src, obj_state_params.unsqueeze(0))
    # #     mask: torch.tensor = kornia.warp_affine(torch.tensor(rand_d_s), M[:, :2, :], dsize=(96, 96))
    # #     mask_rep = torch.repeat_interleave(mask, 4, dim=0).reshape(4, 96, 96)
    # #     mask_rep_np = mask_rep.detach().numpy()
    # #
    # #     mask_w: torch.tensor = kornia.warp_affine(torch.ones(*rand_d_s.shape), M[:, :2, :], dsize=(96, 96))
    # #     mask_w_rep = torch.repeat_interleave(mask_w, 4, dim=0).reshape(4, 96, 96)
    # #     mask_w_rep_np = mask_w_rep.detach().numpy()
    # #
    # #     rand_score += reward
    # #     state = state_
    # #     if done or die:
    # #         break
    # #
    # # print('Ep {}\tScore: {:.2f}\t'.format(1, rand_score))
    #
    # print('Running attack')
    # env = Env(args.seed, args.img_stack, args.action_repeat)
    # # import time
    # # time.sleep(10)
    # for i_ep in range(1):
    #     score = 0
    #     state = env.reset()
    #     mask_rep_np = np.zeros_like(state)
    #     mask_w_rep_np = np.zeros_like(state)
    #
    #     i = 0
    #     for t in range(1000):
    #         if t >= 17 * 8:
    #             if i == T:
    #                 attack_tT = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
    #                 break
    #             if i < T:
    #                 if i == 0:
    #                     obs = state
    #                     warped_obs = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
    #                     attack_t0 = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
    #                 state = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
    #                 i += 1
    #                 state.clip(-1, 0.9921875)
    #                 import matplotlib.pyplot as plt
    #
    #                 plt.imshow(state[0], cmap='gray')
    #                 plt.pause(0.000000000001)
    #                 # plt.show()
    #                 if i == 1 or i == T:
    #                     plt.show()
    #                 # plt.show()
    #                 plt.clf()
    #         if args.render:
    #             env.render()
    #         action = agent.select_action(state, device)
    #         print(t, action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
    #         state_, reward, done, die, _, car_props, obj_poly = env.step(
    #             action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
    #
    #         (x, y), angle, (linVelx, linVely) = car_props[0], car_props[1], car_props[2]
    #         obj_state_params = get_obj_params(torch.tensor([x, y, angle, linVelx, linVely]), obj_true_loc)
    #
    #         points_src = torch.tensor([[[0., 0.], [d_s_size_y, 0.], [d_s_size_y, d_s_size_x], [0., d_s_size_x]]])
    #         M: torch.tensor = kornia.get_perspective_transform(points_src, obj_state_params.unsqueeze(0))
    #         # if t == 0:
    #         #     M: torch.tensor = kornia.get_perspective_transform(points_src, obj_state_params.unsqueeze(0))
    #         #     M_tmp = M
    #         # else:
    #         #     M = M_tmp
    #         # mask: torch.tensor = kornia.warp_perspective(torch.tensor(d_s), M, dsize=(96, 96))
    #         mask: torch.tensor = kornia.warp_affine(torch.tensor(d_s), M[:, :2, :], dsize=(96, 96))
    #         # mask.clamp_(-eps, eps)
    #         # plt.imshow(mask.squeeze().detach().numpy(), cmap='gray')
    #         # plt.show()
    #         mask_rep = torch.repeat_interleave(mask, 4, dim=0).reshape(4, 96, 96)
    #         mask_rep_np = mask_rep.detach().numpy()
    #
    #         mask_w: torch.tensor = kornia.warp_affine(torch.ones(*d_s.shape), M[:, :2, :], dsize=(96, 96))
    #         mask_w_rep = torch.repeat_interleave(mask_w, 4, dim=0).reshape(4, 96, 96)
    #         mask_w_rep_np = mask_w_rep.detach().numpy()
    #
    #         score += reward
    #         state = state_
    #         if done or die:
    #             break
    #
    #     print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
    #
    # # # for diagram
    # # plt.imshow(obs[3], cmap='gray')
    # # plt.axis('off')
    # # plt.show()
    # # plt.savefig('animations/diagram/obs.png')
    # #
    # # plt.imshow(np.random.randn(25, 25), cmap='gray')
    # # plt.axis('off')
    # # plt.show()
    # # plt.savefig('animations/diagram/patch.png')
    # #
    # # plt.imshow(warped_obs[3], cmap='gray')
    # # plt.axis('off')
    # # plt.show()
    # # plt.savefig('animations/diagram/warped_obs.png')
    # #
    # # plt.imshow(obs[3], cmap='gray')
    # # plt.axis('off')
    # # plt.show()
    # # plt.savefig('animations/diagram/next_obs.png')
    # #
    # # plt.imshow(target_obs[3], cmap='gray')
    # # plt.axis('off')
    # # plt.show()
    # # plt.savefig('animations/diagram/target.png')
    #
    # # intro_path = 'animations'
    # # # for intro
    # # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # # axes[0].imshow(attack_t0[3], cmap='gray')
    # # axes[0].axis('off')
    # # axes[1].imshow(attack_tT[3], cmap='gray')
    # # axes[1].axis('off')
    # # fig.savefig(f'{intro_path}/intro_full.png', bbox_inches='tight')
    #
    # # scenario_path = 'animations/scenario3'
    # # # for scenarios
    # # plt.imshow(target_obs[3], cmap='gray')
    # # plt.axis('off')
    # # plt.show()
    # # plt.savefig(f'{scenario_path}/target.png', bbox_inches='tight')
    # #
    # # plt.imshow(start_obs[3], cmap='gray')
    # # plt.axis('off')
    # # plt.show()
    # # plt.savefig(f'{scenario_path}/start_obs.png', bbox_inches='tight')
    # #
    # # plt.imshow(random_t0[3], cmap='gray')
    # # plt.axis('off')
    # # plt.show()
    # # plt.savefig(f'{scenario_path}/random_t0.png', bbox_inches='tight')
    # #
    # # plt.imshow(random_tT[3], cmap='gray')
    # # plt.axis('off')
    # # plt.show()
    # # plt.savefig(f'{scenario_path}/random_tT.png', bbox_inches='tight')
    # #
    # # plt.imshow(attack_t0[3], cmap='gray')
    # # plt.axis('off')
    # # plt.show()
    # # plt.savefig(f'{scenario_path}/attack_t0.png', bbox_inches='tight')
    # #
    # # plt.imshow(attack_tT[3], cmap='gray')
    # # plt.axis('off')
    # # plt.show()
    # # plt.savefig(f'{scenario_path}/attack_tT.png', bbox_inches='tight')
