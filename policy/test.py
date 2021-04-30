import argparse
import numpy as np
import torch

from policy.env.env import Env
from policy.agents.agent import Agent

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

if __name__ == "__main__":
    agent = Agent(args.img_stack, device)
    agent.load_param(device)
    env = Env(args.seed, args.img_stack, args.action_repeat)

    training_records = []
    running_score = []
    state = env.reset()
    no_of_rollouts = 100
    for i_ep in range(no_of_rollouts):
        score = 0
        state = env.reset()

        for t in range(1000):
            action = agent.select_action(state, device)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            score += reward
            state = state_
            if done or die:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
        running_score.append(score)

    print(f'Total average reward score: {np.mean(running_score)}')
    print(f'Total standard deviation reward score: {np.std(running_score)}')