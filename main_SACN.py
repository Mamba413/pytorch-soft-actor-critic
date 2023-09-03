import argparse
import gym
import d4rl
import numpy as np
import torch
from sacN import SAC_N
from pathlib import Path
from util import set_seed, Log, sample_batch, evaluate_policy
from util import get_env_and_heavy_tail_dataset

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="BENCH",
                    help='BENCH or MM or Mean')
parser.add_argument('--agent_policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--N', type=int, default=5,
                    help='the number of ensembles')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--eval-period', type=int, default=5000)
parser.add_argument('--n-eval-episodes', type=int, default=10)
parser.add_argument('--max-episode-steps', type=int, default=1000)
parser.add_argument('--log-dir', type=str, default='NONE')
parser.add_argument("--k-fold", default=5, type=int)
parser.add_argument('--quantile', type=float, default=0.5)
parser.add_argument('--scale', type=float, default=2.0)
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

if __name__ == '__main__':
    if args.policy == "BENCH":
        K_FOLDS = 1
    else:
        K_FOLDS = args.k_fold
    
    if args.log_dir == "NONE":
        args.log_dir = args.policy
        if args.policy == "MM":
            args.log_dir = args.log_dir + "_q_{}".format(args.quantile)
        if args.policy == "PB":
            args.log_dir = args.log_dir + "_std_{}".format(args.scale)
    log = Log(Path(args.log_dir)/args.env_name/'seed_{}'.format(args.seed), vars(args))
    log(f'Log dir: {log.dir}')

    # Environment and dataset
    env, dataset = get_env_and_heavy_tail_dataset(log, args.env_name, args.max_episode_steps, 12345*args.seed, df=args.df, std_noise=args.std_noise)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(args.seed, env=env)

    # Agent
    agent = SAC_N(obs_dim, act_dim, args)
    def eval_policy():
        eval_returns = np.array([evaluate_policy(env, agent, args.max_episode_steps) for _ in range(args.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        log.row({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
            'normalized return mean': normalized_returns.mean(),
            'normalized return std': normalized_returns.std(),
        })

    # Update agent
    for step in range(args.num_steps):
        agent.update_parameters(updates=step, **sample_batch(dataset, args.batch_size))
        if (step+1) % args.eval_period == 0:
            eval_policy()

    env.close()
    log.close()