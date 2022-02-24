"""
Encapsulate generate data to make it parallel
"""
import argparse
import os
import sys
from multiprocessing import Pool
from os import makedirs
from os.path import join
from subprocess import call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--scenario', type=str, default='straight', help='Select driving scenario')
parser.add_argument('--same-track', action='store_true', help="Generates same track for every rollout if specified")
parser.add_argument('--rollouts', type=int, help="Total number of rollouts.")
parser.add_argument('--file', type=str, default="data_collection.carracing", help="File which need to run parallely")
parser.add_argument('--threads', type=int, default=1, help="Number of threads")
parser.add_argument('--rootdir', type=str, help="Directory to store rollout ")
parser.add_argument('--policy', type=str, choices=['pre', 'pre_noise', 'random_1', 'random_2'],
                    help="Directory to store rollout directories of each thread",
                    default='pre')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args()

if args.scenario not in ['straight', 'left_turn', 'right_turn']:
    raise ValueError(
        f'Invalid scenario {args.scenario}. Please choose scenario either straight, left_turn or right_turn')

rpt = args.rollouts // args.threads + 1
if args.rollouts % args.threads == 0:
    rpt -= 1
root_dir = f'{args.rootdir}/scenario_{args.scenario}'


def _threaded_generation(i):
    tdir = join(root_dir, '{}_thread_{}'.format(args.policy, i))
    makedirs(tdir, exist_ok=True)
    cmd = ["python", "-m", args.file, "--dir",
           tdir, "--rollouts", str(rpt), "--policy", args.policy, "--scenario", str(args.scenario)]
    if args.same_track:
        cmd += ["--same-track"]
    if args.render:
        cmd += ["--render"]
    cmd = " ".join(cmd)
    print(cmd)
    call(cmd, shell=True)
    return True


with Pool(args.threads) as p:
    p.map(_threaded_generation, range(args.threads))
