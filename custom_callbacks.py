#custom_callbacks.py>

import gymnasium as gym
import aero_gym
from stable_baselines3 import TD3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure, HParam
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from typing import Any, Dict

class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def _on_training_start(self) -> None:
        print("calling hparamcallback")
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "batch size": self.model.batch_size,
            "buffer size": self.model.buffer_size,
            "learning starts": self.model.learning_starts,
        }
        if type(self.model) is DQN:
            hparam_dict.update(
                {
                    "exploration fraction": self.model.exploration_fraction,
                    "exploration initial_eps": self.model.exploration_initial_eps,
                    "exploration final_eps": self.model.exploration_final_eps,
                }
            )
        if type(self.model) is TD3:
            if type(self.model.action_noise) is NormalActionNoise:
                hparam_dict.update(
                    {
                        "(first) action noise mean": float(self.model.action_noise._mu[0]),
                        "(first) action noise sigma": float(self.model.action_noise._sigma[0]),
                    }
                )
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

class FigureRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, log_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a figure of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param log_freq: Render the agent's trajectory every log_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self._log_freq = log_freq
        self._n_calls = 0
        
    def _on_step(self) -> bool:
        return True
        
    def _on_rollout_start(self) -> bool:
        self._n_calls += 1
        if self.model.num_timesteps > self.model.learning_starts and self._n_calls % self._log_freq == 0:
            print("########################## Recording figure ##########################")
            alpha_eff = []
            alpha_dot = []
            alpha_ddot = []
            h_ddot = []
            fy = []
            t = []
            no_action_fy = []
            no_action_t = []
            no_action_alpha_eff = []
            am_action_fy = []
            am_action_t = []
            am_action_alpha_eff = []
            am_action_alpha_dot = []
            am_action_alpha_ddot = []

            def collect_data(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Collect the current time, states and previous lift and add to arrays for plotting

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
#                 print(_locals)
#                 alpha_eff.append(_locals["observations"][-1][-4])
#                 alpha_dot.append(_locals["observations"][-1][-3])
                alpha_eff.append(_locals["infos"][-1]["current alpha_eff"])
                alpha_dot.append(_locals["infos"][-1]["current alpha_dot"])
                alpha_ddot.append(_locals["infos"][-1]["previous alpha_ddot"])
                h_ddot.append(_locals["infos"][-1]["current h_ddot"])
                fy.append(_locals["infos"][-1]["previous fy"])
                t.append(_locals["infos"][-1]["current t"])

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=collect_data,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )

            t_max = self.model.env.envs[0].unwrapped.t_max
            delta_t = self.model.env.envs[0].unwrapped.delta_t
            h_ddot_list = np.copy(h_ddot)
            # Pad with zeros such that there are enough entries to go to t_max
            h_ddot_list = np.pad(h_ddot_list, (0, int(t_max / delta_t) + 1 - len(h_ddot_list)), 'constant')

            no_action_env = gym.make(
                'aero_gym/wagner_jones-v0', 
                render_mode="ansi", 
                t_max=t_max, 
                delta_t=delta_t, 
                continuous_actions=True,
                lift_threshold=1,
                alpha_ddot_threshold=0.0)

            am_action_env = gym.make(
                'aero_gym/wagner_jones-v0', 
                render_mode="ansi", 
                t_max=t_max, 
                delta_t=delta_t, 
                continuous_actions=True,
                lift_threshold=1)
            
            
            no_action_env.reset(options={"h_ddot_prescribed": h_ddot_list})
            done = False
            while done is False:
                _, _, _, done, info = no_action_env.step([0.0])
                no_action_fy.append(info["previous fy"])
                no_action_t.append(info["current t"])
                no_action_alpha_eff.append(info["current alpha_eff"])
                
            am_action = 1 / am_action_env.alpha_ddot_threshold * np.gradient(h_ddot_list, delta_t)
            am_action_env.reset(options={"h_ddot_prescribed": h_ddot_list})

            done = False
            i = 0
            while done is False:
                action = am_action[i]
                _, _, _, done, info = am_action_env.step([action])
                am_action_fy.append(info["previous fy"])
                am_action_t.append(info["current t"])
                am_action_alpha_eff.append(info["current alpha_eff"])
                am_action_alpha_dot.append(info["current alpha_dot"])
                am_action_alpha_ddot.append(info["previous alpha_ddot"])
                i += 1

            am_action_color = 'red'
            no_action_color = 'orange'
            policy_color = 'blue'
            plt.rcParams['lines.linewidth'] = 1
    
            lift_threshold = self.model.env.envs[0].unwrapped.lift_threshold
            figure, axarr = plt.subplots(ncols=3, nrows=2, figsize=(17,7), )
            axarr[0,0].set_ylabel('lift')
            axarr[0,1].set_ylabel('h dot')
            axarr[0,2].set_ylabel('h ddot')
            axarr[1,0].set_ylabel('alpha (eff)')
            axarr[1,1].set_ylabel('alpha dot')
            axarr[1,2].set_ylabel('alpha ddot')
            
            axarr[0,0].set_ylim([-2*lift_threshold,2*lift_threshold])
            axarr[0,0].plot(t[:-1],fy[1:],marker='o',markersize=2,label=f"Current policy", color=policy_color)
            axarr[0,0].plot(no_action_t[:-1],no_action_fy[1:],marker='o',markersize=2,label=f"No action", color=no_action_color)
            axarr[0,0].plot(am_action_t[:-1],am_action_fy[1:],marker='o',markersize=2,label=f"AM compensation",color=am_action_color)
            axarr[0,0].hlines([-lift_threshold,lift_threshold],no_action_t[:-1][0],no_action_t[:-1][-1],linestyles='dashed',colors='black')
            
            axarr[0,2].plot(t[:-1],h_ddot[:-1],marker='o',markersize=2, color=policy_color)
            axarr[0,2].plot(no_action_t[:-1],h_ddot_list[:-2],marker='o',markersize=2, color=no_action_color)
            
            axarr[1,0].set_ylim([-0.003,0.003])
            axarr[1,0].plot(t[:-1],alpha_eff[:-1],marker='o',markersize=2, color=policy_color)
            axarr[1,0].plot(am_action_t[:-1],am_action_alpha_eff[:-1],marker='o',markersize=2, color=am_action_color)
            axarr[1,0].plot(no_action_t[:-1],no_action_alpha_eff[:-1],marker='o',markersize=2, color=no_action_color)
            
            axarr[1,1].plot(t[:-1],alpha_dot[:-1],marker='o',markersize=2, color=policy_color)
            axarr[1,1].plot(am_action_t[:-1],am_action_alpha_dot[:-1],marker='o',markersize=2, color=am_action_color)
            
            axarr[1,2].plot(t[:-1],alpha_ddot[1:],marker='o',markersize=2, color=policy_color)
            axarr[1,2].plot(am_action_t[:-1],am_action_alpha_ddot[1:],marker='o',markersize=2, color=am_action_color)

            
            axarr[0,0].legend()
            for ax in axarr.flatten():
                ax.set_xlabel('time')
                ax.set_xlim([0.0,t_max])
    
            self.logger.record(
                "trajectory/figure", 
                Figure(figure, close=True), 
                exclude=("stdout", "log", "json", "csv")
            )
            plt.close()

        return True
