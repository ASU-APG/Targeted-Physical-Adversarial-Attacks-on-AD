import argparse
from os import mkdir
from os.path import exists

import kornia
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from data_collection.utils import scenarios_start_pos, scenarios_seeds, scenarios_object_points
from gym.envs.box2d.car_racing_adv import WINDOW_W, WINDOW_H, STATE_W, STATE_H
from policy.agents.agent import Agent
from policy.env.env_adv import Env
from utils import PERTURBATION_SIZE, get_perturbation_file_path, get_target_state

parser = argparse.ArgumentParser(description='Test different scenarios of physical adversarial attacks on CarRacing-v0')
parser.add_argument('--scenario', type=str, default='straight', help='select driving scenario')
parser.add_argument('--action-repeat', type=int, default=1, metavar='N', help='repeat action in N frames (default: 1)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--unroll-length', type=int, default=25, help='Number of timesteps to attack')
parser.add_argument('--adv-bound', type=float, default=0.9, help='Adversarial bound for perturbation')
parser.add_argument('--targets-dir', type=str, default='attacks/targets',
                    help="directory where target states exist")
parser.add_argument('--perturbs-dir', type=str, default='attacks/perturbations',
                    help="directory where perturbations are saved")
parser.add_argument('--results-dir', type=str, default='results',
                    help="directory where results need to be saved")
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--save', action='store_true', help='save the results if specified')
args = parser.parse_args()

# cuda and seeds
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
if use_cuda:
    torch.cuda.manual_seed(0)

# results save directory
save_dir = f'{args.results_dir}/scenario_{args.scenario}'
if not exists(save_dir):
    mkdir(save_dir)

# common variables
d_s_size_x, d_s_size_y = PERTURBATION_SIZE[0], PERTURBATION_SIZE[1]
obj_true_loc = scenarios_object_points[args.scenario]


def get_obj_params_robust(car_props, object_true_loc, robust_vec):
    zoom = 16.200000000000003
    x = car_props[0].unsqueeze(0).unsqueeze(0)
    y = car_props[1].unsqueeze(0).unsqueeze(0)
    angle = car_props[2]
    linVelx = car_props[3]
    linVely = car_props[4]

    object_true_loc = [(x - robust_vec[0].item(), y - robust_vec[1].item()) for (x, y) in object_true_loc]

    angle = -angle
    if torch.norm(torch.tensor([linVelx, linVely])) > 0.5:
        angle = torch.atan2(linVelx, linVely)
    obj_state_params = None
    for idx in range(len(object_true_loc)):
        tmp_x = object_true_loc[idx][0] - x
        tmp_y = object_true_loc[idx][1] - y
        obj_x = zoom * (tmp_x * torch.cos(angle) - tmp_y * torch.sin(angle)) / WINDOW_W * STATE_W + STATE_W / 2
        obj_y = zoom * (tmp_x * torch.sin(angle) + tmp_y * torch.cos(angle)) / WINDOW_H * STATE_H + STATE_H / 4
        if idx == 0:
            obj_state_params = torch.cat([obj_x, obj_y], dim=1)
        else:
            obj_state_params = torch.cat([obj_state_params, torch.cat([obj_x, obj_y], dim=1)])

    obj_state_params -= torch.tensor([0., 96.], device=device)
    obj_state_params[:, 1] = -obj_state_params[:, 1]
    return obj_state_params


def run_env_robust(env, agent, d_s, target, start_pos, T, robust_vec=torch.tensor([0, 0])):
    t0, tT = None, None
    actions = []
    reward_subset = 0
    loss_target = 0

    perturb = d_s

    state, _ = env.reset()
    mask_rep_np = np.zeros_like(state)
    mask_w_rep_np = np.zeros_like(state)

    i = 0
    score = 0
    for t in range(1000):
        if t >= start_pos:
            if i == T:
                tT = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
                break
            if i < T:
                if i == 0:
                    t0 = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
                state = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
                i += 1
                state.clip(-1, 0.9921875)
        if args.render:
            env.render()
        action = agent.select_action(state, device)
        action = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
        state_, reward, done, die, _, car_props, obj_poly, _ = env.step(action)
        if 1 <= i <= T:
            actions.append(action)
            reward_subset += reward
            loss_target += np.sum((target - state_) ** 2) / (4 * 96 * 96)

        (x, y), angle, (linVelx, linVely) = car_props[0], car_props[1], car_props[2]
        obj_state_params = get_obj_params_robust(torch.tensor([x, y, angle, linVelx, linVely]), obj_true_loc,
                                                 robust_vec)

        points_src = torch.tensor([[[0., 0.], [d_s_size_y, 0.], [d_s_size_y, d_s_size_x], [0., d_s_size_x]]])
        M: torch.tensor = kornia.get_perspective_transform(points_src, obj_state_params.unsqueeze(0))
        mask: torch.tensor = kornia.warp_affine(torch.tensor(perturb.copy()), M[:, :2, :], dsize=(96, 96))
        mask_rep = torch.repeat_interleave(mask, 4, dim=0).reshape(4, 96, 96)
        mask_rep_np = mask_rep.detach().numpy()

        mask_w: torch.tensor = kornia.warp_affine(torch.ones(*perturb.shape), M[:, :2, :], dsize=(96, 96))
        mask_w_rep = torch.repeat_interleave(mask_w, 4, dim=0).reshape(4, 96, 96)
        mask_w_rep_np = mask_w_rep.detach().numpy()

        score += reward
        state = state_
        if done or die:
            break

    print('Score: {:.2f}\t'.format(score))
    return t0, tT, actions, reward_subset, loss_target / T


def run_robustness_test(env, agent, d_s, target, limits, start_pos, T):
    x_m_dir, x_p_dir = limits[0]
    y_m_dir, y_p_dir = limits[1]

    heatmap_mat = np.zeros((x_p_dir - x_m_dir + 1, y_p_dir - y_m_dir + 1))
    i, j = 0, 0

    for x in range(x_m_dir, x_p_dir + 1):
        j = 0
        for y in range(y_m_dir, y_p_dir + 1):
            _, _, _, _, s_loss_attack = run_env_robust(env, agent, d_s, target, start_pos, T, torch.tensor([x, y]))
            heatmap_mat[i][j] = s_loss_attack
            j += 1
        i += 1

    return heatmap_mat


def plot_robustness(heatmap_mat, limits, T, eps, save):
    if save:
        matplotlib.use('Agg')

    (x_m_dir, x_p_dir), (y_m_dir, y_p_dir) = limits

    x_dir = list(reversed(-np.arange(x_m_dir, x_p_dir + 1)))
    y_dir = list(reversed(-np.arange(y_m_dir, y_p_dir + 1)))

    heatmap_mat = np.flip(heatmap_mat)

    fig, ax = plt.subplots()
    # im = ax.imshow(heatmap_mat, cmap='Reds_r')
    im = ax.imshow(heatmap_mat, interpolation='none', cmap=plt.get_cmap('winter'))

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Attack Loss', rotation=-90, va="bottom", fontsize=15)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y_dir)))
    ax.set_yticks(np.arange(len(x_dir)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y_dir)
    ax.set_yticklabels(x_dir)

    ax.set_xlabel('X Direction', fontsize=15)
    ax.set_ylabel('Y Direction', fontsize=15)

    fig_name = f'robustness_heatmap_T_{T}_eps_{eps}'
    if not save:
        ax.set_title(fig_name)
    plt.show()
    if save:
        plt.savefig(f'{save_dir}/{fig_name}.png', bbox_inches='tight')
    plt.close()


def main():
    # attack parameters
    T = args.unroll_length
    eps = args.adv_bound
    # perturbation file name to be fetched
    perturb_file = get_perturbation_file_path(args.perturbs_dir, args.scenario, T, eps)
    d_s = np.load(perturb_file)['arr_0']
    # get target state
    target = get_target_state(args.targets_dir, args.scenario)
    # start position for each scenario
    start_pos = scenarios_start_pos[args.scenario] * 8  # 8 accounts for policy action repeat
    if args.scenario == 'straight':
        start_pos -= 1  # to be even more precise
    # environment seeds for each scenario
    env_seed = scenarios_seeds[args.scenario]

    # init agent
    agent = Agent(args.img_stack, device)
    agent.load_param(device)

    # init env
    env = Env(env_seed, args.img_stack, args.action_repeat, args.scenario)

    # run policy on env in the presence of attack
    _, _, _, _, s_loss_attack_start = run_env_robust(env, agent, d_s, target, start_pos, T)

    # run robustness experiment
    x_m_dir, x_p_dir = -9, 2  # exceeding this can make box go out of track
    y_m_dir, y_p_dir = -9, 2
    heatmap_mat = run_robustness_test(env, agent, d_s, target, [(x_m_dir, x_p_dir), (y_m_dir, y_p_dir)], start_pos, T)
    plot_robustness(heatmap_mat, [(x_m_dir, x_p_dir), (y_m_dir, y_p_dir)], T, eps, args.save)
    print(f'Robustness plot showed/saved for scenario {args.scenario}, T = {T}, and eps = {eps}')


if __name__ == "__main__":
    main()
