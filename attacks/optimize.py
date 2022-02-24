import argparse
import warnings
from os import getcwd
from os.path import exists
import sys
sys.path.append(getcwd())

import kornia
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from torch.distributions.categorical import Categorical

import gym
from data_collection.utils import scenarios_object_points, scenarios_start_pos, scenarios_seeds
from dynamics_model.models.mdrnn import MDRNNCell
from dynamics_model.models.vae import VAE
from gym.envs.box2d.car_racing_adv import WINDOW_W, WINDOW_H, STATE_W, STATE_H, PLAYFIELD
from policy.networks.actor_critic import A2CNet
from utils import PERTURBATION_SIZE, get_target_state, get_perturbation_file_path, save_loss_to_file

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--scenario', type=str,
                    default='straight', help='select driving scenario')
parser.add_argument('--dyn-params-dir', type=str, default='dynamics_model/params',
                    help="directory where dynamics model params exist")
parser.add_argument('--policy-params-dir', type=str, default='policy/param',
                    help="directory where dynamics model params exist")
parser.add_argument('--targets-dir', type=str, default='attacks/targets',
                    help="directory where target states exist")
parser.add_argument('--perturbs-dir', type=str, default='attacks/perturbations',
                    help="directory where perturbations need to be saved")
parser.add_argument('--adv-bound', type=float,
                    default=0.9, help="adversarial bound")
parser.add_argument('--unroll-length', type=int, default=25,
                    help="unroll length for attack")
parser.add_argument('--epochs', type=int, default=1000,
                    help="number of epochs to optimize")
parser.add_argument('--lr', type=float, default=0.005,
                    help="learning rate for optimization")
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--no-load-perturb', action='store_true',
                    help='doesnt load existing perturbation')
args = parser.parse_args()

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running on ', device)


def load_nets():
    # load state vae, rnn
    vae = VAE(3, 32).float().to(device)
    rnn = MDRNNCell(32, 3, 256, 5).float().to(device)
    vae_weights_file = f'{args.dyn_params_dir}/vae_scenario_{args.scenario}/best.tar'
    rnn_weights_file = f'{args.dyn_params_dir}/mdrnn_scenario_{args.scenario}/best.tar'
    # rnn_weights_file = 'dynamics/param/combine_rnn/state_rnn/best_epochs_50_lr_0.001.tar'
    # load vae
    vae_state = torch.load(
        vae_weights_file, map_location=lambda storage, location: storage)
    print("Loading VAE at epoch {}, with test error {}...".format(
        vae_state['epoch'], vae_state['precision']))
    vae.load_state_dict(vae_state['state_dict'])
    decoder = vae.decoder
    # load rnn
    rnn_state = torch.load(
        rnn_weights_file, map_location=lambda storage, location: storage)
    print("Loading MDRNN at epoch {}, with test error {}...".format(
        rnn_state['epoch'], rnn_state['precision']))
    rnn_state_dict = {k.strip('_l0'): v for k,
                      v in rnn_state['state_dict'].items()}
    rnn.load_state_dict(rnn_state_dict)
    # load A2C
    a2c = A2CNet(4).float().to(device)
    a2c_weights_file = f'{args.policy_params_dir}/ppo_net_params.pkl'
    a2c.load_state_dict(torch.load(
        a2c_weights_file, map_location=lambda storage, location: storage))

    # eval
    vae.eval()
    decoder.eval()
    rnn.eval()
    a2c.eval()
    return vae, decoder, rnn, a2c


def main():
    # load networks
    vae, decoder, rnn, a2c = load_nets()
    # set up parameters
    adv_bound = args.adv_bound
    unroll_length = args.unroll_length
    lr = args.lr
    epochs = args.epochs

    # load target state
    s_t = get_target_state(args.targets_dir, args.scenario)
    s_t = torch.repeat_interleave(torch.tensor(s_t[3], device=device)
                                  .reshape(1, 96, 96), 4, dim=0).float().reshape(1, 4, 96, 96)

    # set up perturbation
    d_s_size_x, d_s_size_y = PERTURBATION_SIZE[0], PERTURBATION_SIZE[1]
    d_s = torch.randn(1, 1, d_s_size_x, d_s_size_y,
                      device=device, requires_grad=True)
    # d_s.data = d_s.data.clamp_(0, 1)
    d_s.data = d_s.data.clamp_(-adv_bound, adv_bound)
    best_loss = float('inf')
    # perturbation file name to be saved
    perturb_file = get_perturbation_file_path(
        args.perturbs_dir, args.scenario, unroll_length, adv_bound)

    # load already existing perturbation if we want. Useful for optimizing with breaks
    if not args.no_load_perturb and exists(perturb_file):
        print('Loading previous best perturbation', perturb_file)
        d_s = np.load(perturb_file)['arr_0']
        d_s = torch.tensor(d_s.copy(), device=device, requires_grad=True)
        d_s.data = d_s.data.clamp_(-adv_bound, adv_bound)

    # init optimizer and scheduler
    adam_optim = optim.Adam([d_s], lr=lr)

    # saves first state generation mixture to generate sample states for each epoch
    mixt_list = []

    # select start_l_s, start_s and start_car_props
    env = gym.make("CarRacingAdv-v0", scenario=args.scenario)

    # car props setup
    zoom = 16.200000000000003
    obj_true_loc = scenarios_object_points[args.scenario]
    # 8 accounts for policy action repeat
    start_pos = scenarios_start_pos[args.scenario] * 8
    if args.scenario == 'straight':
        start_pos -= 1  # to be even more precise
    env_seed = scenarios_seeds[args.scenario]

    # not sure why this is here. But still keeping
    torch.autograd.set_detect_anomaly(True)

    # optimize the perturbation for attack
    for i in range(epochs):
        t_loss = 0
        obj_state_params_list = []

        env.seed(env_seed)
        start_s = env.reset()
        if args.render:
            env.render()

        with torch.no_grad():
            start_s = torch.tensor(start_s.copy(), device=device).float(
            ).unsqueeze(0).permute(0, 3, 1, 2) / 255
            mu, logsigma = vae.encoder(start_s)
            sigma = logsigma.exp()
            eps = torch.randn_like(sigma)
            start_l_s = eps.mul(sigma).add_(mu).float()
        start_s = decoder(start_l_s)
        # reshape
        start_s = start_s.clamp(0, 1) * 255
        start_s = start_s.permute(0, 2, 3, 1).contiguous().squeeze()
        assert start_s.shape == (96, 96, 3)
        # stack start state
        start_s = torch.matmul(start_s[..., :], torch.tensor(
            [0.299, 0.587, 0.114], device=device)).reshape(1, 96, 96)
        start_s = start_s / 128. - 1.
        start_s = torch.repeat_interleave(
            start_s, 4, dim=0).reshape(1, 4, 96, 96)

        # this is to get desired start state by wasting running agent in environment for some time
        # these are handpicked as of now
        start_car_props = None
        for elapse in range(start_pos):
            with torch.no_grad():
                (alpha, beta), _ = a2c(start_s)
                action = alpha / (alpha + beta)
                # scale action
                action = action * \
                    torch.tensor([2., 1., 1.], device=device) + \
                    torch.tensor([-1., 0., 0.], device=device)

            state_, reward, done, die, _, car_props, obj_poly = env.step(
                action.squeeze().cpu().numpy())
            if args.render:
                env.render()
            start_s = state_
            with torch.no_grad():
                start_s = torch.tensor(start_s.copy(), device=device).float(
                ).unsqueeze(0).permute(0, 3, 1, 2) / 255
                mu, logsigma = vae.encoder(start_s)
                sigma = logsigma.exp()
                eps = torch.randn_like(sigma)
                start_l_s = eps.mul(sigma).add_(mu).float()
            start_s = torch.tensor(state_.copy(), device=device).float(
            ).unsqueeze(0).permute(0, 3, 1, 2) / 255
            # reshape
            start_s = start_s.clamp(0, 1) * 255
            start_s = start_s.permute(0, 2, 3, 1).contiguous().squeeze()
            assert start_s.shape == (96, 96, 3)
            # stack start state
            start_s = torch.matmul(start_s[..., :], torch.tensor([0.299, 0.587, 0.114], device=device)).reshape(1, 96,
                                                                                                                96)
            start_s = start_s / 128. - 1.
            start_s = torch.repeat_interleave(
                start_s, 4, dim=0).reshape(1, 4, 96, 96)

            # car props setup
            (x, y), angle, (linVelx, linVely), _ = car_props
            x = x / PLAYFIELD
            y = y / PLAYFIELD
            angle = angle / np.pi
            start_car_props = torch.tensor([x, y, angle, linVelx, linVely], device=device).unsqueeze(0).unsqueeze(
                0).float()

        # hidden setup
        h_state = 2 * [torch.zeros((1, 256), device=device)]

        s = start_s
        l_s = start_l_s
        c = start_car_props
        s_h = h_state

        for t in range(unroll_length):
            # get object parameters from car_props
            x = c[:, :, 0] * PLAYFIELD
            y = c[:, :, 1] * PLAYFIELD
            angle = c[:, :, 2] * np.pi
            linVelx = c[:, :, 3]
            linVely = c[:, :, 4]
            angle = -angle
            if torch.norm(torch.tensor([linVelx, linVely], device=device)) > 0.5:
                angle = torch.atan2(linVelx, linVely)
            obj_state_params = None
            for idx in range(len(obj_true_loc)):
                tmp_x = obj_true_loc[idx][0] - x
                tmp_y = obj_true_loc[idx][1] - y
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
            obj_state_params = obj_state_params.to(device)
            obj_state_params_list.append(obj_state_params)
            if len(obj_state_params_list) > unroll_length:
                obj_state_params_list.pop(0)

            points_src = torch.tensor([[[0., 0.], [d_s_size_y, 0.], [d_s_size_y, d_s_size_x], [0., d_s_size_x]]],
                                      device=device)
            M: torch.tensor = kornia.get_perspective_transform(
                points_src, obj_state_params.unsqueeze(0))
            # create mask image from perturbation through warping
            mask: torch.tensor = kornia.warp_affine(
                d_s.float(), M[:, :2, :], dsize=(96, 96), flags='nearest')
            mask.clamp_(-adv_bound, adv_bound)
            mask_rep = torch.repeat_interleave(mask, 4, dim=0) \
                .reshape(1, 4, 96, 96)

            # set up mask of white
            mask_w: torch.tensor = kornia.warp_affine(
                torch.ones_like(d_s).float(), M[:, :2, :], dsize=(96, 96))
            mask_rep_w = torch.repeat_interleave(mask_w, 4, dim=0) \
                .reshape(1, 4, 96, 96)

            if args.render:
                # # commenting this as it's taking too much time. Better save and create animation during test
                # plt.imshow(torch.clamp(s + mask_rep, -1, 0.9921875)[0][0].detach().numpy(), cmap='gray')
                # plt.title(f't = {t}')
                # plt.pause(0.0001)
                pass

            # get adv action from controller
            (alpha, beta), _ = a2c(
                F.tanh(s * (1 - mask_rep_w) + mask_rep_w * mask_rep))
            a = alpha / (alpha + beta)
            # scale action
            a = a * torch.tensor([2., 1., 1.], device=device) + \
                torch.tensor([-1., 0., 0.], device=device)

            mu, sigma, pi, r, d, hidden_s = rnn(a, l_s, s_h)
            pi = pi.squeeze()
            mixt = Categorical(torch.exp(pi)).sample().item()
            # Ensures that mixt is sampled such that it stays the same every epoch
            if len(mixt_list) < unroll_length:
                mixt_list.append(mixt)
                l_s_ = mu[:, mixt, :]
            else:
                l_s_ = mu[:, mixt_list[t], :]

            # get next state
            s_ = decoder(l_s_)
            # reshape
            s_ = s_.clamp(0, 1) * 255
            s_ = s_.permute(0, 2, 3, 1).contiguous().squeeze()
            assert s_.shape == (96, 96, 3)
            # stack next state
            s_ = torch.matmul(s_[..., :], torch.tensor(
                [0.299, 0.587, 0.114], device=device)).reshape(1, 96, 96)
            s_ = s_ / 128. - 1.
            s_ = torch.repeat_interleave(s_, 4, dim=0).reshape(1, 4, 96, 96)
            # get and add loss
            loss = F.mse_loss(s_, s_t)
            t_loss += loss

            # pick next state, next latent, next car_props from environment
            state_, reward, done, die, _, car_props, obj_poly = env.step(
                a.detach().squeeze().cpu().numpy())
            if args.render:
                env.render()

            with torch.no_grad():
                start_s = torch.tensor(state_.copy(), device=device).float(
                ).unsqueeze(0).permute(0, 3, 1, 2) / 255
                mu, logsigma = vae.encoder(start_s)
                sigma = logsigma.exp()
                eps = torch.randn_like(sigma)
                l_s = eps.mul(sigma).add_(mu).float()
            # start_s setup
            tmp = torch.tensor(state_.copy(), device=device).float(
            ).unsqueeze(0).permute(0, 3, 1, 2) / 255
            # reshape
            tmp = tmp.clamp(0, 1) * 255
            tmp = tmp.permute(0, 2, 3, 1).contiguous().squeeze()
            assert tmp.shape == (96, 96, 3)
            tmp = torch.matmul(tmp[..., :], torch.tensor(
                [0.299, 0.587, 0.114], device=device)).reshape(1, 96, 96)
            tmp = tmp / 128. - 1.
            tmp = torch.repeat_interleave(tmp, 4, dim=0).reshape(1, 4, 96, 96)
            s.data = tmp.data

            (x, y), angle, (linVelx, linVely), _ = car_props
            x = x / PLAYFIELD
            y = y / PLAYFIELD
            angle = angle / np.pi
            tmp = torch.tensor([x, y, angle, linVelx, linVely], device=device).unsqueeze(
                0).unsqueeze(0).float()
            c.data = tmp.data

            s_h = hidden_s

        adam_optim.zero_grad()
        t_loss.backward(retain_graph=True)

        adam_optim.step()

        with torch.no_grad():
            for t in range(unroll_length):
                pass
            d_s.clamp_(-adv_bound, adv_bound)

        print(
            f'Epoch: {i + 1}/{epochs} | Loss: {t_loss.item() / unroll_length}')

        if t_loss < best_loss:
            best_loss = t_loss
            print(f'best d_s found at epoch {i + 1}. Saving ...')
            np.savez_compressed(perturb_file, d_s.detach().cpu().numpy())
            save_loss_to_file(t_loss.item(
            ) / unroll_length, args.perturbs_dir, args.scenario, unroll_length, adv_bound)


if __name__ == '__main__':
    main()
