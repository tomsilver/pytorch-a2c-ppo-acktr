import argparse
import os

import numpy as np
import torch

import gym_residual_grasping
from gym import wrappers, logger

from envs import VecPyTorch, make_vec_envs, make_env
from utils import get_render_func, get_vec_normalize


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det

env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                            None, None, args.add_timestep, device='cpu',
                            allow_early_resets=False)
env.render = env.venv.venv.envs[0].render

# env = make_env(args.env_name, args.seed + 1000, 1, None, args.add_timestep, False)()

# Get a render function
# import pdb; pdb.set_trace()
# render_func = env.venv.venv.envs[0].render # get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

logger.set_level(logger.INFO)

outdir = '/tmp/trained-agent-results'
env = wrappers.Monitor(env, directory=outdir, force=True, video_callable=lambda episode_id: True)
env.seed(0)

episode_count = 30
done = False

for i in range(episode_count):
    print("episode", i)
    obs = env.reset()
    episode_reward = 0

    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # print("aciton:", action.shape)
        # import pdb; pdb.set_trace()

        # Obser reward and next obs
        obs, reward, done, _ = env.step(np.squeeze(action))
        episode_reward += reward
        if done:
            print("episode reward:", episode_reward)
            break

