import argparse
from os import mkdir, getcwd
from os.path import exists

import kornia
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from data_collection.utils import scenarios_start_pos, scenarios_seeds, scenarios_object_points
from gym.envs.box2d.car_racing_adv import WINDOW_W, WINDOW_H, STATE_W, STATE_H
from policy.agents.agent import Agent
from policy.env.env_adv import Env
from utils import PERTURBATION_SIZE, get_perturbation_file_path, get_target_state, \
    create_animation_video, read_loss_from_file

parser = argparse.ArgumentParser(
    description='Test different scenarios of physical adversarial attacks on CarRacing-v0')
parser.add_argument('--scenario', type=str,
                    default='straight', help='select driving scenario')
parser.add_argument('--action-repeat', type=int, default=1,
                    metavar='N', help='repeat action in N frames (default: 1)')
parser.add_argument('--img-stack', type=int, default=4,
                    metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--unroll-length', type=int, default=25,
                    help='Number of timesteps to attack')
parser.add_argument('--adv-bound', type=float, default=0.9,
                    help='Adversarial bound for perturbation')
parser.add_argument('--targets-dir', type=str, default='attacks/targets',
                    help="directory where target states exist")
parser.add_argument('--perturbs-dir', type=str, default='attacks/perturbations',
                    help="directory where perturbations are saved")
parser.add_argument('--results-dir', type=str, default='results',
                    help="directory where results need to be saved")
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--save', action='store_true',
                    help='save the results if specified')
args = parser.parse_args()

if args.scenario != 'straight':
    args.scenario += '_turn'
    
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


def get_obj_params(car_props, object_true_loc):
    """
    To get object parameters from car dynamics and given true object location
    :param car_props:
    :param object_true_loc:
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
    for idx in range(len(object_true_loc)):
        tmp_x = object_true_loc[idx][0] - x
        tmp_y = object_true_loc[idx][1] - y
        obj_x = zoom * (tmp_x * torch.cos(angle) - tmp_y *
                        torch.sin(angle)) / WINDOW_W * STATE_W + STATE_W / 2
        obj_y = zoom * (tmp_x * torch.sin(angle) + tmp_y *
                        torch.cos(angle)) / WINDOW_H * STATE_H + STATE_H / 4
        if idx == 0:
            obj_state_params = torch.cat([obj_x, obj_y], dim=1)
        else:
            obj_state_params = torch.cat(
                [obj_state_params, torch.cat([obj_x, obj_y], dim=1)])

    obj_state_params -= torch.tensor([0., 96.], device=device)
    obj_state_params[:, 1] = -obj_state_params[:, 1]
    return obj_state_params


def run_env_no_object(env, agent, start_pos, T):
    t0, tT = None, None
    actions = []
    state, _ = env.reset()
    reward_subset = 0
    agent_states = []

    states_list = []

    i = 0
    score = 0
    for t in range(1000):
        if t >= start_pos:
            if i == T:
                tT = state
                break
            if i < T:
                if i == 0:
                    t0 = state
                i += 1
            states_list.append(state)
        if args.render:
            env.render()
        action = agent.select_action(state, device)
        action = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
        state_, reward, done, die, _, car_props, obj_poly, _ = env.step(action)
        if 1 <= i <= T:
            actions.append(action)
            reward_subset += reward
            (x, y), angle, (linVelx,
                            linVely), omega = car_props[0], car_props[1], car_props[2], car_props[3]
            agent_states.append([x, y, angle, linVelx, linVely])
        score += reward
        state = state_
        if done or die:
            break

    print('Score: {:.2f}\t'.format(score))
    return t0, tT, actions, reward_subset, agent_states, states_list


def run_env(env, agent, d_s, start_pos, T):
    t0, tT = None, None
    t0_rgb, tT_rgb = None, None
    actions = []
    reward_subset = 0
    agent_states = []
    states_list = []

    perturb = d_s

    state, state_rgb = env.reset()
    mask_rep_np = np.zeros_like(state)
    mask_w_rep_np = np.zeros_like(state)

    i = 0
    score = 0
    for t in range(1000):
        if t >= start_pos:
            if i == T:
                tT = state * (1 - mask_w_rep_np) + mask_w_rep_np * mask_rep_np
                tT_rgb = state_rgb / 255. * (1 - np.moveaxis(mask_w_rep_np[1:], 0, -1)) + \
                    np.moveaxis(mask_w_rep_np[1:], 0, -1) * \
                    np.moveaxis(mask_rep_np[1:], 0, -1)
                break
            if i < T:
                if i == 0:
                    t0 = state * (1 - mask_w_rep_np) + \
                        mask_w_rep_np * mask_rep_np
                    t0_rgb = state_rgb / 255. * (1 - np.moveaxis(mask_w_rep_np[1:], 0, -1)) + \
                        np.moveaxis(
                            mask_w_rep_np[1:], 0, -1) * np.moveaxis(mask_rep_np[1:], 0, -1)
                state = state * (1 - mask_w_rep_np) + \
                    mask_w_rep_np * mask_rep_np
                i += 1
                state.clip(-1, 0.9921875)
            states_list.append(state)
        if args.render:
            env.render()
        action = agent.select_action(state, device)
        action = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
        state_, reward, done, die, _, car_props, obj_poly, state_rgb = env.step(
            action)
        if 1 <= i <= T:
            actions.append(action)
            reward_subset += reward
            (x, y), angle, (linVelx,
                            linVely), omega = car_props[0], car_props[1], car_props[2], car_props[3]
            agent_states.append([x, y, angle, linVelx, linVely])

        (x, y), angle, (linVelx,
                        linVely) = car_props[0], car_props[1], car_props[2]
        obj_state_params = get_obj_params(torch.tensor(
            [x, y, angle, linVelx, linVely]), obj_true_loc)

        points_src = torch.tensor(
            [[[0., 0.], [d_s_size_y, 0.], [d_s_size_y, d_s_size_x], [0., d_s_size_x]]])
        M: torch.tensor = kornia.get_perspective_transform(
            points_src, obj_state_params.unsqueeze(0))
        mask: torch.tensor = kornia.warp_affine(
            torch.tensor(perturb.copy()), M[:, :2, :], dsize=(96, 96))
        mask_rep = torch.repeat_interleave(mask, 4, dim=0).reshape(4, 96, 96)
        mask_rep_np = mask_rep.detach().numpy()

        mask_w: torch.tensor = kornia.warp_affine(
            torch.ones(*perturb.shape), M[:, :2, :], dsize=(96, 96))
        mask_w_rep = torch.repeat_interleave(
            mask_w, 4, dim=0).reshape(4, 96, 96)
        mask_w_rep_np = mask_w_rep.detach().numpy()

        score += reward
        state = state_
        if done or die:
            break

    print('Score: {:.2f}\t'.format(score))
    return t0, tT, actions, reward_subset, agent_states, states_list, [t0_rgb, tT_rgb]


def run_env_multi_random(env, agent, n_runs, start_pos, T):
    agent_states_multi = []

    for i_ep in range(n_runs):
        perturb = np.random.normal(0, 0.3, size=(
            1, 1, d_s_size_x, d_s_size_y)).astype(np.float32)
        state, _ = env.reset()
        mask_rep_np = np.zeros_like(state)
        mask_w_rep_np = np.zeros_like(state)
        agent_states = []
        i = 0
        score = 0
        for t in range(1000):
            if t >= start_pos:
                if i == T:
                    tT = state * (1 - mask_w_rep_np) + \
                        mask_w_rep_np * mask_rep_np
                    break
                if i < T:
                    if i == 0:
                        t0 = state * (1 - mask_w_rep_np) + \
                            mask_w_rep_np * mask_rep_np
                    state = state * (1 - mask_w_rep_np) + \
                        mask_w_rep_np * mask_rep_np
                    i += 1
                    state.clip(-1, 0.9921875)
            if args.render:
                env.render()
            action = agent.select_action(state, device)
            action = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
            state_, reward, done, die, _, car_props, obj_poly, _ = env.step(
                action)
            if 1 <= i <= T:
                (x, y), angle, (linVelx,
                                linVely), omega = car_props[0], car_props[1], car_props[2], car_props[3]
                agent_states.append([x, y, angle, linVelx, linVely])

            (x, y), angle, (linVelx,
                            linVely) = car_props[0], car_props[1], car_props[2]
            obj_state_params = get_obj_params(torch.tensor(
                [x, y, angle, linVelx, linVely]), obj_true_loc)

            points_src = torch.tensor(
                [[[0., 0.], [d_s_size_y, 0.], [d_s_size_y, d_s_size_x], [0., d_s_size_x]]])
            M: torch.tensor = kornia.get_perspective_transform(
                points_src, obj_state_params.unsqueeze(0))
            mask: torch.tensor = kornia.warp_affine(
                torch.tensor(perturb.copy()), M[:, :2, :], dsize=(96, 96))
            mask_rep = torch.repeat_interleave(
                mask, 4, dim=0).reshape(4, 96, 96)
            mask_rep_np = mask_rep.detach().numpy()

            mask_w: torch.tensor = kornia.warp_affine(
                torch.ones(*perturb.shape), M[:, :2, :], dsize=(96, 96))
            mask_w_rep = torch.repeat_interleave(
                mask_w, 4, dim=0).reshape(4, 96, 96)
            mask_w_rep_np = mask_w_rep.detach().numpy()

            score += reward
            state = state_
            if done or die:
                break

        agent_states_multi.append(np.array(agent_states))
        print('Score: {:.2f}\t'.format(score))

    agent_states_multi_mean = np.mean(agent_states_multi, axis=0)
    agent_states_multi_std = np.std(agent_states_multi, axis=0)
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


def plot_intro_image(fig_list, fig_name, save):
    if save:
        matplotlib.use('Agg')
    # for intro
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(fig_list[0])
    axes[0].axis('off')
    axes[1].imshow(fig_list[1])
    axes[1].axis('off')
    if not save:
        plt.title(fig_name)
    plt.show()
    if save:
        fig.savefig(f'{save_dir}/{fig_name}.png', bbox_inches='tight')
    plt.close()


def plot_actions(actions_list, names, T, eps, save):
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
        plt.savefig(f'{save_dir}/steer_T_{T}_eps_{eps}.png',
                    bbox_inches='tight')
    plt.close()


def plot_trajectories(a_state_list, names, std, T, eps, save):
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
        plt.plot(positions[:, 0], positions[:, 1],
                 colors[idx], transform=rot + base, lw=4)
        # The below does a std curve about mean. But negligible to draw
        # if idx == 1:
        #     plt.plot(positions[:, 0] - std[:, 0], positions[:, 1] - std[:, 1], colors[idx].replace('--', ':'),
        #              transform=rot + base, lw=1, label='_nolegend_')
        #     plt.plot(positions[:, 0] + std[:, 0], positions[:, 1] + std[:, 1], colors[idx].replace('--', ':'),
        #              transform=rot + base, lw=1, label='_nolegend_')
        if idx == 0:
            plt.scatter(positions[:1, 0], positions[:1, 1],
                        c='black', s=200, transform=rot + base)
        if idx == 2:
            plt.scatter(positions[-1:, 0], positions[-1:, 1], c=colors[idx].replace('--', ''), marker='*', s=200,
                        transform=rot + base)
        else:
            plt.scatter(positions[-1:, 0], positions[-1:, 1], c=colors[idx].replace('--', ''), transform=rot + base,
                        label='_nolegend_')
        plt.tick_params(axis=u'both', which=u'both', length=0)
        plt.xlabel('X Position', fontsize=20)
        plt.ylabel('Y Position', fontsize=20)
        plt.xlim(-26, 15)
    plt.legend(names + ['Start', 'Target'],
               loc='lower left', prop={'size': 18})
    plt.grid(alpha=0.5)
    plt.show()
    if save:
        plt.savefig(f'{save_dir}/trajectory_T_{T}_eps_{eps}.png',
                    bbox_inches='tight')
    plt.close()


def main():
    # attack parameters
    T = args.unroll_length
    eps = args.adv_bound
    # get random and adversarial perturbation
    rand_d_s = np.random.uniform(
        low=-eps, high=eps, size=(1, 1, d_s_size_x, d_s_size_y)).astype(np.float32)
    # perturbation file name to be fetched
    perturb_file = get_perturbation_file_path(
        args.perturbs_dir, args.scenario, T, eps)
    d_s = np.load(perturb_file)['arr_0']
    # get target state
    target = get_target_state(args.targets_dir, args.scenario)
    # start position for each scenario
    # 8 accounts for policy action repeat
    start_pos = scenarios_start_pos[args.scenario] * 8
    if args.scenario == 'straight':
        start_pos -= 1  # to be even more precise
    # environment seeds for each scenario
    env_seed = scenarios_seeds[args.scenario]

    # init agent
    agent = Agent(args.img_stack, device)
    agent.load_param(device)

    # init env
    env = Env(env_seed, args.img_stack, args.action_repeat, args.scenario)

    # show or save target state
    plot_images([target], ['target'], args.save)
    print(
        f'Target state showed/saved for scenario {args.scenario}, T = {T}, and eps = {eps}')

    # run policy on env without object
    s_0, s_t, a_no_obj, rew_no_obj, a_state_no_obj, states_no_obj = run_env_no_object(
        env, agent, start_pos, T)
    plot_images([s_0, s_t], ['start_no_obj', 'end_no_obj'], args.save)
    print(
        f'Plot without object showed/saved for scenario {args.scenario}, T = {T}, and eps = {eps}')

    # run policy on env in the presence of attack
    attack_s_0, attack_s_t, a_attack, rew_attack, a_states_attack, states_attack, s_rgb = run_env(env, agent, d_s,
                                                                                                  start_pos, T)
    plot_images([attack_s_0, attack_s_t], [
                f'start_attack_T_{T}_eps_{eps}', f'end_attack_T_{T}_eps_{eps}'], args.save)
    print(
        f'Plot with adversarial object showed/saved for scenario {args.scenario}, T = {T}, and eps = {eps}')

    # plot intro image
    plot_intro_image(s_rgb, f'intro_T_{T}_eps_{eps}', args.save)
    print(
        f'Intro image showed/saved for scenario {args.scenario}, T = {T}, and eps = {eps}')

    # animate video
    create_animation_video(save_dir, states_no_obj, states_attack, T, eps)
    print(
        f'Animation video saved for scenario {args.scenario}, T = {T}, and eps = {eps}')

    # run policy on env in the presence of random noise
    # This is to show images with random baseline
    rand_s_0, rand_s_t, a_rand, rew_rand, a_states_rand, states_random, _ = run_env(
        env, agent, rand_d_s, start_pos, T)
    plot_images([rand_s_0, rand_s_t], [
                'start_random', 'end_random'], args.save)
    print(
        f'Plot with random perturbation showed/saved for scenario {args.scenario}, T = {T}, and eps = {eps}')

    # run multiple random and get average
    # This is to show trajectories based on mean of multiple random baseline
    a_states_rand_mean, a_states_rand_std = run_env_multi_random(
        env, agent, 10, start_pos, T)

    # plot trajectories using agent states
    plot_trajectories([a_state_no_obj, a_states_rand_mean, a_states_attack],
                      ['No Object', 'Random Noise', 'Attack'], a_states_rand_std, T, eps, args.save)
    print(
        f'Different trajectories plotted for scenario {args.scenario}, T = {T}, and eps = {eps}')

    # # plot steering angle. Not shown in paper. But good to have
    # plot_actions([a_no_obj, a_rand, a_attack], ['No Object', 'Random Noise', 'Attack'], T, eps, args.save)
    # print(f'Different steering angles plotted for scenario {args.scenario}, T = {T}, and eps = {eps}')

    # actions error
    print('======================== Actions MSE ========================')
    a_err_rand = np.linalg.norm(
        (np.array(a_no_obj) - np.array(a_rand) ** 2) / T)
    a_err_attack = np.linalg.norm(
        (np.array(a_no_obj) - np.array(a_attack) ** 2) / T)
    print(
        f'Scenario {args.scenario}, T = {T}, and eps = {eps}, Action error random noise {a_err_rand}')
    print(
        f'Scenario {args.scenario}, T = {T}, and eps = {eps}, Action error attack {a_err_attack}')

    # reward reduction
    print('======================== Reward Change ========================')
    reward_red_random = (rew_rand - rew_no_obj) * 100 / rew_no_obj
    reward_red_attack = (rew_attack - rew_no_obj) * 100 / rew_no_obj
    print(
        f'Scenario {args.scenario}, T = {T}, and eps = {eps}, Reward reduction random noise {reward_red_random}')
    print(
        f'Scenario {args.scenario}, T = {T}, and eps = {eps}, Reward reduction attack {reward_red_attack}')

    # attack loss. May not be present for all scenarios
    attack_loss = read_loss_from_file(args.perturbs_dir, args.scenario, T, eps)
    if attack_loss:
        print('======================== Attack Loss ========================')
        print(
            f'Scenario {args.scenario}, T = {T}, and eps = {eps}, Attack loss {attack_loss}')


if __name__ == "__main__":
    main()
