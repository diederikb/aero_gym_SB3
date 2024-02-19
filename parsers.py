# parsers.py

import argparse
import logging
import json
import numpy as np
from copy import deepcopy
from stable_baselines3.common.noise import NormalActionNoise
from trajectory_generators import *

def parse_training_args(cli_args, general_input_as_dict={}, env_input_as_dict={}):
    # Use command-line input to get the input file and to overwrite any settings in the input file
    cli_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    cli_env_parser = argparse.ArgumentParser(parents=[cli_parser], add_help=False, argument_default=argparse.SUPPRESS)

    cli_parser.add_argument("--input_file", type=str, help="json file with case parameters")
    add_general_cli_args(cli_parser)
    add_training_cli_args(cli_parser)
    add_wrapper_cli_args(cli_parser)
    add_env_cli_args(cli_env_parser)

    cli_general_args, unknown_args = cli_parser.parse_known_args(cli_args)
    cli_env_args = cli_env_parser.parse_args(unknown_args)
    logging.basicConfig(level=cli_general_args.loglevel)

    # Create dicts from arguments
    cli_general_dict = vars(cli_general_args)
    cli_env_dict = vars(cli_env_args)

    # Assemble input_dict in this order:
    # - first parse input json file (if it exists)
    # - overwrite with cli_args
    # - overwrite with general_input_as_dict
    input_dict = {}
    if "input_file" in cli_general_dict.keys():
        with open(cli_general_args.input_file) as jf:
            input_file_dict = json.load(jf)
            input_dict.update(input_file_dict)
    input_dict.update(cli_general_dict)
    input_dict.update(general_input_as_dict)

    # Assemble input_dict["env_kwargs"] in this order:
    # - first use "env_kwargs" from input json file (if it exists)
    # - overwrite with cli_args
    # - overwrite with env_input_as_dict
    input_env_dict = {}
    if "env_kwargs" in input_dict:
        input_env_dict = deepcopy(input_dict["env_kwargs"])
    input_env_dict.update(cli_env_dict)
    input_env_dict.update(env_input_as_dict)
    input_dict["env_kwargs"] = input_env_dict

    # Create defaults for some keys if they are not present in parsed_input_dict
    general_defaults = {
            "model_restart_dir": None,
            "model_restart_file": None,
        }
    env_defaults = {
            "h_ddot_generator": "constant(0)",
            "reference_lift_generator": "constant(0)",
            "sys_reinit_commands": None,
            "observe_vorticity_field": False
        }
    for k, v in general_defaults.items():
        if k not in input_dict:
            input_dict[k] = v
    for k, v in env_defaults.items():
        if k not in input_dict["env_kwargs"]:
            input_dict["env_kwargs"][k] = v

    # Generate values that should be objects, but that are saved as strings in the json file
    parsed_input_dict = deepcopy(input_dict)
    for k in ["train_freq", "action_noise"]:
        if k in parsed_input_dict["training_algorithm_kwargs"].keys():
            parsed_input_dict["training_algorithm_kwargs"][k] = eval(input_dict["training_algorithm_kwargs"][k])
    for k in ["h_ddot_generator", "reference_lift_generator"]:
        if k in parsed_input_dict["env_kwargs"].keys():
            parsed_input_dict["env_kwargs"][k] = eval(input_dict["env_kwargs"][k])

    return parsed_input_dict, input_dict

def parse_eval_args(cli_args, general_input_as_dict={}, env_input_as_dict={}):
    # Use command-line input to get the input file and to overwrite any settings in the input file
    cli_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    cli_env_parser = argparse.ArgumentParser(parents=[cli_parser], add_help=False, argument_default=argparse.SUPPRESS)

    cli_parser.add_argument("--input_file", type=str, help="json file with case parameters")
    add_general_cli_args(cli_parser)
    add_eval_cli_args(cli_parser)
    add_wrapper_cli_args(cli_parser)
    add_env_cli_args(cli_env_parser)

    cli_general_args, unknown_args = cli_parser.parse_known_args(cli_args)
    cli_env_args = cli_env_parser.parse_args(unknown_args)
    logging.basicConfig(level=cli_general_args.loglevel)

    # Create dicts from arguments
    cli_general_dict = vars(cli_general_args)
    cli_env_dict = vars(cli_env_args)

    # Assemble input_dict in this order:
    # - first parse input json file (if it exists)
    # - overwrite with cli_args
    # - overwrite with general_input_as_dict
    input_dict = {}
    if "input_file" in cli_general_dict.keys():
        with open(cli_general_args.input_file) as jf:
            input_file_dict = json.load(jf)
            input_dict.update(input_file_dict)
    input_dict.update(cli_general_dict)
    input_dict.update(general_input_as_dict)

    # Assemble input_dict["env_kwargs"] in this order:
    # - first use "env_kwargs" from input json file (if it exists)
    # - overwrite with cli_args
    # - overwrite with env_input_as_dict
    input_env_dict = {}
    if "env_kwargs" in input_dict:
        input_env_dict = deepcopy(input_dict["env_kwargs"])
    input_env_dict.update(cli_env_dict)
    input_env_dict.update(env_input_as_dict)
    input_dict["env_kwargs"] = input_env_dict

    # Create defaults for some keys if they are not present in parsed_input_dict
    general_defaults = {
            "archive_dir": "archive",
            "archive_prefix": "",
            "start_at": 0,
            "end_at": np.inf,
        }
    env_defaults = {
            "h_ddot_generator": "constant(0)",
            "reference_lift_generator": "constant(0)",
            "sys_reinit_commands": None,
            "observe_vorticity_field": False
        }
    for k, v in general_defaults.items():
        if k not in input_dict:
            input_dict[k] = v
    for k, v in env_defaults.items():
        if k not in input_dict["env_kwargs"]:
            input_dict["env_kwargs"][k] = v

    # Generate values that should be objects, but that are saved as strings in the json file
    parsed_input_dict = deepcopy(input_dict)
    for k in ["h_ddot_generator", "reference_lift_generator"]:
        if k in parsed_input_dict["env_kwargs"].keys():
            parsed_input_dict["env_kwargs"][k] = eval(input_dict["env_kwargs"][k])

    return parsed_input_dict, input_dict

def add_general_cli_args(parser):
    """
    Add CLI arguments that can always be used.
    """
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )

def add_wrapper_cli_args(parser):
    """
    Add CLI arguments specific for Gymnasium/StableBaselines wrappers.
    """
    parser.add_argument("--stacked_frames", type=int, help="number of frames used for the SB3 VecFrameStack wrapper")

def add_training_cli_args(parser):
    """
    Add CLI arguments specific for training.
    """
    parser.add_argument("--root_dir", type = str, help="root directory where to put case_name dirs for results/logging files")
    parser.add_argument("--case_name", type=str, help="name for the directory where to put results/logging files")
    parser.add_argument("--algorithm", type=str, help="learning algorithm")
    parser.add_argument("--policy", type=str, help="the policy model to use")
    parser.add_argument("--net_arch", type=int, nargs='*', help="network layer sizes")
    parser.add_argument("--total_timesteps", type=int, help="total training timesteps")
    parser.add_argument("--model_restart_file", type=str, help="if specified, load this model and restart the training")
    parser.add_argument("--model_restart_dir", type=str, help="if specified, load the latest model in this directory and restart the training, or, if there is no saved model, start a new one")

def add_eval_cli_args(parser):
    """
    Add CLI arguments specific for evaluation.
    """
    parser.add_argument("--eval_dir", type=str, help="directory containing the models to be evaluated")
    parser.add_argument("--archive_dir", type=str, help="directory within eval_dir where evaluations are saved")
    parser.add_argument("--archive_prefix", type=str, help="prefix for npz archive file name")
    parser.add_argument("--algorithm", type=str, help="learning algorithm")
    parser.add_argument("--eval_freq", type=int, help="if specified, only evaluate at timestep multiples of this value")
    parser.add_argument("--n_eval_episodes", type=int, help="if specified, only evaluate at timestep multiples of this value")
    parser.add_argument("--start_at", type=int, help="only evaluate models with timesteps higher or equal to this value")
    parser.add_argument("--end_at", type=int, help="only evaluate models with timesteps below or equal to this value")
    parser.add_argument("--archive_to_overwrite", type=str, help="if specified, overwrite values in this archive")

def add_env_cli_args(parser):
    """
    Add CLI arguments specific for AeroGym environments to the provided parser.
    """
    parser.add_argument("--delta_t", type=float, help="minimum time step of the RL environment")
    parser.add_argument("--t_max", type=float, help="maximum episode time length")
    parser.add_argument("--reward_type", type=int, help="reward type")
    parser.add_argument("--xlim", type=float, nargs='*', help="x limits of the flow domain")
    parser.add_argument("--ylim", type=float, nargs='*', help="y limits of the flow domain")
    parser.add_argument("--initialization_time", type=float, help="the time after impulsively starting the flow for the solution that is used to initialize the episodes")
    parser.add_argument("--alpha_init", type=float, help="the initial angle of attack in radians (used also for the flow initialization). Note that the observed alpha is relative to this one")
    parser.add_argument('--observe_wake', action='store_true')
    parser.add_argument('--observe_vorticity_field', action='store_true')
    parser.add_argument('--observe_previous_wake', action='store_true')
    parser.add_argument('--observe_alpha', action='store_true')
    parser.add_argument('--observe_alpha_dot', action='store_true')
    parser.add_argument('--observe_alpha_ddot', action='store_true')
    parser.add_argument('--observe_h_ddot', action='store_true')
    parser.add_argument('--observe_h_dot', action='store_true')
    parser.add_argument('--observe_alpha_eff', action='store_true')
    parser.add_argument('--observe_previous_alpha_eff', action='store_true')
    parser.add_argument('--observe_circulation', action='store_true')
    parser.add_argument('--observe_previous_lift', action='store_true')
    parser.add_argument('--observe_previous_lift_error', action='store_true')
    parser.add_argument('--observe_previous_lift_sqrt_error', action='store_true')
    parser.add_argument('--observe_previous_lift_integrated_error', action='store_true')
    parser.add_argument('--observe_previous_circulatory_pressure', action='store_true')
    parser.add_argument('--observe_previous_pressure', action='store_true')
    parser.add_argument("--lift_termination", action=argparse.BooleanOptionalAction, help="terminate the episode if the unscaled lift exceeds lift_upper_limit or subceeds lift_lower_limit")
    parser.add_argument("--alpha_termination", action=argparse.BooleanOptionalAction, help="terminate the episode if the unscaled lift exceeds alpha_upper_limit or subceeds alpha_lower_limit")
    parser.add_argument("--alpha_dot_termination", action=argparse.BooleanOptionalAction, help="terminate the episode if the absolute value of the unscaled alpha_dot exceeds alpha_dot_limit")
    parser.add_argument("--lift_scale", type=float, help="value that observed lift is scaled by")
    parser.add_argument("--lift_upper_limit", type=float, help="upper limit for the unscaled lift. The episode terminates if exceeded")
    parser.add_argument("--lift_lower_limit", type=float, help="lower limit for the unscaled lift. The episode terminates if exceeded")
    parser.add_argument("--alpha_upper_limit", type=float, help="upper limit for the unscaled AOA. The episode terminates if exceeded")
    parser.add_argument("--alpha_lower_limit", type=float, help="lower limit for the unscaled AOA. The episode terminates if exceeded")
    parser.add_argument("--alpha_dot_limit", type=float, help="limit for the absolute value of angular velocity. The episode terminates if exceeded")
    parser.add_argument("--alpha_ddot_scale", type=float, help="value that input alpha_ddot gets scaled by")
    parser.add_argument("--h_ddot_scale", type=float, help="value that input h_ddot gets scaled by")
    parser.add_argument("--vorticity_scale", type=float, help="value that the vorticity gets scaled by")
    parser.add_argument("--pressure_sensor_positions", type=float, nargs='*', help="x-coordinate of the pressure sensors in the body reference frame")
    parser.add_argument("--reference_lift_generator", type=str, help="reference lift generator used to generate episodes")
    parser.add_argument("--h_ddot_generator", type=str, help="h_ddot generator used to generate episodes")
    parser.add_argument("--sys_reinit_commands", type=str, help="file that contains julia instructions to reinitialize the system at the beginning of every episode")
