import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym import Env, spaces
from offline_rl_algorithms.offline_replay_buffers import H5ReplayBuffer
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, SAC
import torch as th
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to get rid of the warning message
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from typing import Any, Dict

import os
import argparse
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
    CheckpointCallback,
)

# from test_scripts.eval_utils import offline_eval

# from kitchen_env_wrappers import readGif
import imageio
import wandb
from wandb.integration.sb3 import WandbCallback
import io
import random
import torch.nn.functional as F


from offline_rl_algorithms.iql import IQL
from offline_rl_algorithms.rlpd import RLPD

from offline_rl_algorithms.base_offline_rl_algorithm import OfflineRLAlgorithm
from offline_rl_algorithms.wandb_logger import WandBLogger
from offline_rl_algorithms.callbacks import CustomWandbCallback, OfflineEvalCallback
from offline_rl_algorithms.custom_feature_extractors import FlatRangeFeaturesExtractor

# from reward_model.xclip_encoder import XCLIPEncoder
# from reward_model import image_encoders
from reward_model.base_reward_model import BaseRewardModel
from reward_model.rewind_reward_model import ReWiNDRewardModel

from reward_model.env_reward_model import EnvRewardModel
from stable_baselines3.common.logger import configure


from stable_baselines3.common.policies import ActorCriticPolicy
import stable_baselines3


import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


def create_exp_name(cfg: DictConfig):
    exp_name = cfg.environment.cfg_name + "_"

    exp_name += cfg.reward_model.name + "_"

    exp_name += cfg.general_training.name + "_"

    # if cfg.general_training.algo == "iql":
    #     # add policy_extraction and awr/ddpg params
    #     exp_name += f"pe_{cfg.general_training.policy_extraction}_"

    #     if cfg.general_training.policy_extraction == "awr":
    #         exp_name += f"adv_temp_{cfg.general_training.awr_advantage_temp}_"
    #     elif cfg.general_training.policy_extraction == "ddpg":
    #         exp_name += f"bc_weight_{cfg.general_training.ddpg_bc_weight}_"

    #     exp_name += f"n_critics_{cfg.general_training.n_critics}_"
    #     exp_name += f"n_critics_to_sample_{cfg.general_training.n_critics_to_sample}_"
    #     exp_name += f"utd_{cfg.online_training.critic_update_ratio}_"

    # if cfg.general_training.algo == "cql":
    #     exp_name += f"min_q_weight_{cfg.general_training.cql_min_q_weight}_"
    #     exp_name += f"min_q_temp_{cfg.general_training.cql_min_q_temp}_"
    #     exp_name += f"n_critics_{cfg.general_training.n_critics}_"
    #     exp_name += f"n_critics_to_sample_{cfg.general_training.n_critics_to_sample}_"
    #     exp_name += f"utd_{cfg.online_training.critic_update_ratio}_"

    # if cfg.general_training.algo == "rlpd":
    #     exp_name += f"n_critics_{cfg.general_training.n_critics}_"
    #     exp_name += f"n_critics_to_sample_{cfg.general_training.n_critics_to_sample}_"
    #     exp_name += f"train_critic_with_entropy_{cfg.general_training.rlpd_train_critic_with_entropy}_"
    #     exp_name += f"utd_{cfg.online_training.critic_update_ratio}_"

    if cfg.environment.ignore_language:
        exp_name += "no_lang_"

    if cfg.environment.is_state_based:
        exp_name += "state_based_"

    if cfg.environment.use_proprio:
        exp_name += "use_proprio_"

    exp_name += f"action_chunk_{cfg.general_training.action_chunk_size}_Seed_{cfg.general_training.seed}"

    if cfg.reward_model.use_progress_diff:
        exp_name += "_diff"

    # if the last character is an underscore, remove it
    if exp_name[-1] == "_":
        exp_name = exp_name[:-1]

    return exp_name


def parse_reward_model(reward_cfg: DictConfig) -> BaseRewardModel:
    reward_string = reward_cfg.name
    if reward_string is None:
        return None
    if reward_string == "rewind":
        # turn from local to absolute path
        model_path = to_absolute_path(reward_cfg.model_path)
        reward_model = ReWiNDRewardModel(
            model_path,
            camera_names=reward_cfg.camera_names,
            batch_size=reward_cfg.batch_size,
            success_bonus=reward_cfg.success_bonus,
            reward_at_every_step=reward_cfg.reward_at_every_step,
        )
    elif reward_string == "rewind_two_cam":
        # turn from local to absolute path
        from omegaconf import ListConfig

        assert isinstance(reward_cfg.model_path, ListConfig) or isinstance(
            reward_cfg.model_path, list
        )
        model_paths = []
        for i, path in enumerate(reward_cfg.model_path):
            model_paths.append(to_absolute_path(path))
        reward_model = ReWiNDRewardModel(
            model_paths,
            camera_names=reward_cfg.camera_names,
            batch_size=reward_cfg.batch_size,
            success_bonus=reward_cfg.success_bonus,
            reward_at_every_step=reward_cfg.reward_at_every_step,
        )
    elif reward_string == "sparse":
        reward_model = EnvRewardModel(
            reward_type="sparse",
            model_path=reward_cfg.model_path,
            success_bonus=reward_cfg.success_bonus,
        )
    elif reward_string == "dense":
        reward_model = EnvRewardModel(
            reward_type="dense",
            model_path=reward_cfg.model_path,
            success_bonus=reward_cfg.success_bonus,
        )
    elif reward_string == "debug":
        reward_model = EnvRewardModel(
            reward_type="dense",
            model_path=reward_cfg.model_path,
            success_bonus=reward_cfg.success_bonus,
        )
    elif reward_string == "robometer":
        from reward_model.robometer_reward_model import RobometerRewardModel
        reward_model = RobometerRewardModel(
            model_path=reward_cfg.model_path,
            batch_size=reward_cfg.batch_size,
            success_bonus=reward_cfg.success_bonus,
            reward_at_every_step=reward_cfg.reward_at_every_step,
            max_frames=reward_cfg.get("max_frames", 4),
            use_server=reward_cfg.get("use_server", False),
            server_url=reward_cfg.get("server_url", "http://localhost:8000"),
        )
    else:
        raise ValueError(f"Unknown reward model: {reward_string}")

    # Set the success bonus
    reward_model.set_success_bonus(reward_cfg.success_bonus)
    reward_model.set_reward_divisor(reward_cfg.reward_divisor)
    print(f"Success bonus: {reward_model.success_bonus}")

    return reward_model


# Define the function to initialize Hydra
@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Extract configurations
    training_config = cfg.general_training
    env_config = cfg.environment
    model_config = cfg.model
    logging_config = cfg.logging
    offline_config = cfg.offline_training

    experiment_name = os.environ.get("WANDB_RUN_NAME", create_exp_name(cfg))

    ### Setup wandb and logging ###
    if logging_config.wandb:
        config_for_wandb = OmegaConf.to_container(cfg, resolve=True)
        absolute_log_dir = os.path.abspath(logging_config.log_dir)
        print("absolute log dir is", absolute_log_dir)
        config_for_wandb["log_dir"] = absolute_log_dir
        wandb.init(
            entity=logging_config.wandb_entity_name,
            project=logging_config.wandb_project_name,
            group=logging_config.wandb_group_name,
            name=experiment_name,
            config=config_for_wandb,
            monitor_gym=True,
            sync_tensorboard=True,
            notes=cfg.wandb_notes,
        )

    log_dir = logging_config.log_dir

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    reward_model = parse_reward_model(cfg.reward_model)

    wandb_logger = WandBLogger()
    # use a basic logger
    # wandb_logger = configure("logging", [""])
    ### Create environment and callbacks ###
    envs, eval_env = create_envs(cfg, reward_model, logger=wandb_logger)
    model, model_class, policy_kwargs = get_policy_algorithm(
        cfg, envs, log_dir, reward_model
    )

    # Set eval freq and video freq if not set

    # if it's rlpd, video_freq should be never
    # if training_config.algo == "rlpd":
    if "koch" in env_config.cfg_name:
        # if False:
        video_freq = 0
        eval_freq = 0
        # video_freq = offline_config.offline_training_steps * env_config.n_envs // (10)
    else:
        video_freq = offline_config.offline_training_steps * env_config.n_envs // 5
        eval_freq = offline_config.offline_training_steps * env_config.n_envs // (5)

        if logging_config.eval_freq == 0:
            eval_freq = 0

        if logging_config.video_freq == 0:
            video_freq = 0

    # Use deterministic actions for evaluation
    eval_callback = OfflineEvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        video_freq=video_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=25,
    )

    callback_list = generate_callback_list(logging_config, eval_callback)

    # Create the logger
    model.set_logger(wandb_logger)

    use_language = not env_config.ignore_language

    if offline_config.offline_tasks == "all":
        offline_tasks = None
    else:
        offline_tasks = offline_config.offline_tasks

    # Map the tasks to their strings
    # offline_task_strings =

    # model.offline_algo.save("./logs/temp")

    if (
        cfg.offline_training.offline_training_steps > 0
        or cfg.online_training.mix_buffers_ratio > 0
    ) and isinstance(model, OfflineRLAlgorithm):
        debug_koch = False
        if cfg.reward_model.name == "debug" and "koch" in env_config.cfg_name:
            debug_koch = True

        try:
            if (
                cfg.reward_model.name == "rewind_two_cam"
                or cfg.reward_model.name == "debug"
            ):
                cfg.reward_model.name = "rewind"

            h5_path = offline_config.offline_h5_path.format(
                cfg.reward_model.name, cfg.reward_model.reward_at_every_step
            )
            h5_path = to_absolute_path(h5_path)
            print("Offline h5 path:", h5_path)
            from time import sleep

            sleep(1)
            # check if file exists
            with open(h5_path, "r") as f:
                pass
        except FileNotFoundError:
            print(
                "File {} not found. This file likely does not have the correct reward preprocessed.".format(
                    h5_path
                )
            )
            raise FileNotFoundError

        sparse_only = True if reward_model.name == "sparse" else False
        buffer = H5ReplayBuffer(
            h5_path,
            use_language_embeddings=use_language,
            success_bonus=cfg.reward_model.success_bonus,
            sparsify_rewards=sparse_only,
            filter_instructions=offline_tasks,
            reward_model=reward_model,
            is_state_based=env_config.is_state_based,
            use_proprio=env_config.use_proprio,
            calculate_mc_returns=training_config.use_calibrated_q,  # only used for cal-ql
            mc_return_gamma=training_config.gamma,
            dense_rewards_at_end=training_config.dense_rewards_at_end,
            reward_divisor=cfg.reward_model.reward_divisor,
            is_metaworld="metaworld" in env_config.cfg_name,
            normalize_actions_koch="koch" in env_config.cfg_name,
            action_chunk_size=cfg.general_training.action_chunk_size,
            pad_action_chunk_with_last_action=(
                True if "koch" in env_config.cfg_name else False
            ),
            debug_koch=debug_koch,
        )

    ### Learn offline
    if offline_config.offline_training_steps > 0 and isinstance(
        model, OfflineRLAlgorithm
    ):

        if hasattr(training_config, "ckpt_path") and training_config.ckpt_path:
            # Then let's load the model from the ckpt path and continue training
            training_config.ckpt_path = to_absolute_path(training_config.ckpt_path)
            print(f"Loading checkpoint from {training_config.ckpt_path}")

            # Note we are skipping the loading of the offline algo
            model = model.load(
                training_config.ckpt_path, offline_algo=model.offline_algo, env=envs
            )

            model.set_logger(wandb_logger)

        if hasattr(offline_config, "ckpt_path") and offline_config.ckpt_path:
            # convert to absolute path from hydra
            offline_config.ckpt_path = to_absolute_path(offline_config.ckpt_path)
            print(f"Loading checkpoint from {offline_config.ckpt_path}")

            # if rlpd, we do a special load
            if training_config.algo == "rlpd":
                # NOTE: assuming that the offline algo has the same observation space as the env
                print(
                    "Loading offline algo from",
                    offline_config.ckpt_path + "_rlpd_offline",
                )
                if model.offline_algo is not None:
                    offline_algo = model.offline_algo.load(
                        offline_config.ckpt_path + "_rlpd_offline",
                        env=envs,
                        custom_objects={
                            "observation_space": envs.observation_space,
                            "action_space": envs.action_space,
                        },
                        print_system_info=True,
                        load_torch_params_only=True,
                    )

                model = model.load(
                    offline_config.ckpt_path,
                    env=envs,
                    custom_objects={
                        "observation_space": envs.observation_space,
                        "action_space": envs.action_space,
                    },
                    print_system_info=True,
                    load_torch_params_only=True,
                )
                model.set_logger(wandb_logger)
                model.learned_offline = True

                if model.offline_algo is not None:
                    model.offline_algo = offline_algo

                model.set_policies_with_offline()

            else:
                model = model.load(
                    offline_config.ckpt_path,
                    env=envs,
                    custom_objects={
                        "observation_space": envs.observation_space,
                        "action_space": envs.action_space,
                    },
                )

            print("Setting chunk size and all")
            # Replace action chunked buffer again since loading it may not always work
            if cfg.general_training.action_chunk_size > 1:
                model.replace_with_chunked_buffer(
                    cfg.general_training.action_chunk_size,
                    buffer_size=cfg.online_training.total_time_steps,
                    evenly_sample_success=True,
                    ratio=0.5,
                    success_bonus=cfg.reward_model.success_bonus,
                    pad_action_chunk_with_last_action=(
                        True if "koch" in env_config.cfg_name else False
                    ),
                )
                print(model.replay_buffer)
                # model.replay_buffer.sample(10)

            # Various other things to set that don't get set by load
            model.set_logger(wandb_logger)
            model.learned_offline = True
            # Set train_freq, gradient_steps, etc.
            model.train_freq = (
                cfg.environment.train_freq_num,
                cfg.environment.train_freq_type,
            )  # type: ignore[arg-type]
            model.gradient_steps = cfg.online_training.gradient_steps
            if hasattr(cfg, "critic_update_ratio"):
                model.online_critic_update_ratio = (
                    cfg.online_training.critic_update_ratio
                )
                model.offline_critic_update_ratio = (
                    cfg.offline_training.critic_update_ratio
                )
                # setting to online because we are loading offline
                model.current_critic_update_ratio = (
                    cfg.online_training.critic_update_ratio
                )
            model._convert_train_freq()

            if hasattr(cfg, "learning_starts"):
                model.learning_starts = cfg.general_training.learning_starts

        else:
            # checkpoint callback. only save 5 times
            save_freq = int(offline_config.offline_training_steps / 5)

            model.learn_offline(
                offline_replay_buffer=buffer,
                train_steps=offline_config.offline_training_steps,
                callback=callback_list,
                batch_size=256,
            )
            model.set_policies_with_offline()
            # save the model in log_dir/last
            save_dir = os.path.join(log_dir, "last_offline")

            if training_config.algo == "rlpd":
                if hasattr(model, "offline_algo") and model.offline_algo is not None:
                    model.offline_algo.save(save_dir + "_rlpd_offline")
                model.save(save_dir, exclude=["offline_algo"])
            else:
                model.save(save_dir)

            # Model is saved at
            absolute_save_dir = os.path.abspath(save_dir)
            print(f"Model saved at {absolute_save_dir}")

            # Log this absolute_save_dir in wandb
            if logging_config.wandb:
                wandb.run.log({"model_dir": absolute_save_dir})

    # offline_eval(model, reward_model)
    # exit()

    # Set the replay buffer back to the original one
    if (
        isinstance(model, OfflineRLAlgorithm)
        and cfg.online_training.mix_buffers_ratio > 0.0
    ):
        model.set_combined_buffer(buffer, ratio=cfg.online_training.mix_buffers_ratio)
    # reset the last layer of the actor

    ### Learn online ###
    logger = model.logger  # set logger in case
    online_eval_freq = logging_config.eval_freq // env_config.n_envs  # // args.nenvsto
    online_video_freq = logging_config.video_freq // env_config.n_envs
    eval_callback.eval_freq = online_eval_freq
    eval_callback.video_freq = online_video_freq

    if cfg.online_training.warm_start_online_rl:
        model.warm_start_online_rl = True

    if cfg.online_training.total_time_steps > 0:
        # logger only exists for offline algorithms

        online_callback_list = callback_list

        try:
            if isinstance(model, OfflineRLAlgorithm):
                model.learn(
                    total_timesteps=int(cfg.online_training.total_time_steps),
                    callback=online_callback_list,
                    logger=logger,
                    progress_bar=True,
                    parallelize=True if "koch" in env_config.cfg_name else False,
                )
            else:
                model.learn(
                    total_timesteps=int(cfg.online_training.total_time_steps),
                    callback=online_callback_list,
                    progress_bar=True,
                )
        except Exception as e:
            # If crashed, let's save the model
            print("Crashing... Saving model")
            model.save(log_dir)
            print("Model saved at", log_dir)

            # print error
            print(e)

            # show tracebackj
            import traceback

            traceback.print_exc()

    model.save(log_dir)
    print("Done and saved to", log_dir)

    if logging_config.wandb:
        wandb.finish()


def create_envs(cfg: DictConfig, reward_model: BaseRewardModel, logger=None):
    env_config = cfg.environment

    if "metaworld" in env_config.cfg_name:
        from envs.metaworld import (
            create_wrapped_env,
            environment_to_instruction,
        )

        # Extract configuration
        # env_id = env_config.env_id
        # env_id = instruction_to_environment[env_config.text_string]
        # text_instruction = env_config.text_string
        env_id = env_config.env_id
        text_instruction = environment_to_instruction[env_id]

    with th.no_grad():
        # Language features for policy (MiniLM 384-dim)
        policy_lang_feat = reward_model.encode_text_for_policy(
            text_instruction
        ).squeeze()
        
        # For policy input, should use policy encoder's text features, not reward model's
        # Here we use policy_lang_feat as language_features
        lang_feat = policy_lang_feat

    ignore_language = env_config.ignore_language

    # Get use_progress_diff from config (default to False if not specified)
    use_progress_diff = cfg.reward_model.get("use_progress_diff", False)

    if "metaworld" in env_config.cfg_name:
        if env_config.n_envs > 1:
            envs = DummyVecEnv(
                [
                    create_wrapped_env(
                        env_id,
                        reward_model=reward_model,
                        language_features=lang_feat if not ignore_language else None,
                        text_instruction=text_instruction if not ignore_language else None,
                        success_bonus=cfg.reward_model.success_bonus,
                        monitor=True,
                        goal_observable=True,
                        is_state_based=env_config.is_state_based,
                        mode="train",
                        use_proprio=env_config.use_proprio,
                        dense_rewards_at_end=cfg.general_training.dense_rewards_at_end,
                        action_chunk_size=cfg.general_training.action_chunk_size,
                        logger=logger,
                        terminate_on_success=cfg.general_training.terminate_on_success,
                        use_progress_diff=use_progress_diff,
                    )
                    for _ in range(env_config.n_envs)
                ]
            )

            eval_env = DummyVecEnv(
                [
                    create_wrapped_env(
                        env_id,
                        reward_model=reward_model,
                        language_features=lang_feat if not ignore_language else None,
                        text_instruction=text_instruction if not ignore_language else None,
                        success_bonus=cfg.reward_model.success_bonus,
                        monitor=True,
                        goal_observable=True,
                        is_state_based=env_config.is_state_based,
                        mode="eval",
                        use_proprio=env_config.use_proprio,
                        action_chunk_size=cfg.general_training.action_chunk_size,
                        logger=logger,
                        terminate_on_success=cfg.general_training.terminate_on_success,
                        use_progress_diff=use_progress_diff,
                    )
                    for _ in range(1)
                ]
            )  # KitchenEnvDenseOriginalReward(time=True)
        else:
            envs = DummyVecEnv(
                [
                    create_wrapped_env(
                        env_id,
                        reward_model=reward_model,
                        language_features=lang_feat if not ignore_language else None,
                        text_instruction=text_instruction if not ignore_language else None,
                        success_bonus=cfg.reward_model.success_bonus,
                        monitor=True,
                        goal_observable=True,
                        is_state_based=env_config.is_state_based,
                        mode="train",
                        use_proprio=env_config.use_proprio,
                        dense_rewards_at_end=cfg.general_training.dense_rewards_at_end,
                        action_chunk_size=cfg.general_training.action_chunk_size,
                        logger=logger,
                        terminate_on_success=cfg.general_training.terminate_on_success,
                        use_progress_diff=use_progress_diff,
                    )
                ]
            )
            eval_env = DummyVecEnv(
                [
                    create_wrapped_env(
                        env_id,
                        reward_model=reward_model,
                        language_features=lang_feat if not ignore_language else None,
                        text_instruction=text_instruction if not ignore_language else None,
                        success_bonus=cfg.reward_model.success_bonus,
                        monitor=True,
                        goal_observable=True,
                        is_state_based=env_config.is_state_based,
                        mode="eval",
                        use_proprio=env_config.use_proprio,
                        action_chunk_size=cfg.general_training.action_chunk_size,
                        logger=logger,
                        terminate_on_success=cfg.general_training.terminate_on_success,
                        use_progress_diff=use_progress_diff,
                    )
                ]
            )  # KitchenEnvDenseOriginalReward(time=True)
    elif "koch" in env_config.cfg_name:
        camera_kwargs = {
            "image_keys": env_config.image_keys,
            "reward_image_key": env_config.reward_image_key,
        }
        wrapped_env_func = create_wrapped_env(
            env_id,
            language_features=lang_feat,
            policy_language_features=policy_lang_feat if not ignore_language else None,
            reward_model=reward_model,
            goal_observable=True,
            success_bonus=cfg.reward_model.success_bonus,
            is_state_based=env_config.is_state_based,
            use_proprio=env_config.use_proprio,
            mode="train",
            dense_rewards_at_end=cfg.general_training.dense_rewards_at_end,
            action_chunk_size=cfg.general_training.action_chunk_size,
            camera_kwargs=camera_kwargs,
            robot_disabled=env_config.robot_disabled,
            max_episode_steps=env_config.max_episode_steps,
            logger=logger,
        )

        # Define envs (dummy example for illustration)
        if env_config.n_envs > 1:
            envs = SubprocVecEnv([wrapped_env_func for _ in range(env_config.n_envs)])
        else:
            envs = DummyVecEnv([wrapped_env_func])
        eval_env = envs

    return envs, eval_env


def get_policy_algorithm(cfg: DictConfig, envs: VecEnv, log_dir: str, reward_model):
    env_config = cfg.environment
    model_config = cfg.model

    args = cfg.general_training

    if cfg.general_training.action_noise is not None:
        n_actions = envs.action_space.shape[-1]
        action_noise = stable_baselines3.common.noise.NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=cfg.general_training.action_noise * n_actions,
        )
    else:
        action_noise = None

    dim_ranges = []
    projection_dims = []

    if hasattr(envs, "orig_obs_space"):
        orig_obs_space = getattr(envs, "orig_obs_space")
    elif hasattr(envs.envs[0], "orig_obs_space"):  # if vecenv
        orig_obs_space = getattr(envs.envs[0], "orig_obs_space")
    else:
        raise ValueError(
            "envs does not have orig_obs_keys attribute. Use a FlattenDictObservationWrapper"
        )

    orig_obs_keys = orig_obs_space.spaces.keys()

    # if language, it's first
    if "language_feature" in orig_obs_keys:
        dim_ranges.append(orig_obs_space["language_feature"].shape[0])
        projection_dims.append(128)
    # then images (policy must use policy_image_feature)
    policy_image_keys = [key for key in orig_obs_keys if "policy_image_feature" in key]
    
    if policy_image_keys:
        # Policy must use policy encoder's image features
        for key in sorted(policy_image_keys):
            dim_ranges.append(orig_obs_space[key].shape[0])
            projection_dims.append(512)
    else:
        raise KeyError("policy_image_feature_* not found in observation space. Policy requires policy encoder encoded features.")

    # then proprio
    if "proprio" in orig_obs_keys:
        dim_ranges.append(orig_obs_space["proprio"].shape[0])
        projection_dims.append(128)

    policy_kwargs = {
        "net_arch": dict(pi=model_config.pi_net_arch, qf=model_config.qf_net_arch),
        "policy_layer_norm": model_config.policy_layer_norm,
        "critic_layer_norm": model_config.critic_layer_norm,
        "features_extractor_class": FlatRangeFeaturesExtractor,
        "features_extractor_kwargs": {
            "dim_ranges": dim_ranges,
            "projection_dims": projection_dims,
            "normalize_images": True,
        },
    }

    if cfg.model.policy_type == "RnnMlpPolicy":
        policy_kwargs["action_sequence_length"] = cfg.general_training.action_chunk_size

    # everything except BC, SAC, and PPO require n_critics
    if (
        cfg.general_training.algo == "iql"
        or cfg.general_training.algo == "cql"
        or cfg.general_training.algo == "rlpd"
    ):
        policy_kwargs["n_critics"] = cfg.general_training.n_critics

    algo = args.algo.lower()

    # If using RLPD, we instantiate a different offline model
    orig_algo = algo
    if args.algo.lower() == "rlpd":
        algo = cfg.general_training.rlpd_offline_algo.lower()

    if algo == "ppo":
        model_class = PPO
        if not args.pretrained:
            model = model_class(
                "MlpPolicy",
                envs,
                verbose=1,
                tensorboard_log=log_dir,
                n_steps=args.n_steps,
                batch_size=args.n_steps * cfg.environment.n_envs,
                n_epochs=1,
                ent_coef=args.entropy_term,
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    elif algo == "sac":
        model_class = SAC

        # For SAC, we cannot take anything besides net_arch as a parameter
        policy_kwargs = {
            "net_arch": policy_kwargs["net_arch"],
            "features_extractor_class": FlatRangeFeaturesExtractor,
            "features_extractor_kwargs": {
                "dim_ranges": dim_ranges,
                "projection_dims": projection_dims,
                "normalize_images": True,
            },
        }

        if not args.pretrained:
            model = model_class(
                "MlpPolicy",
                envs,
                verbose=1,
                tensorboard_log=log_dir,
                # batch_size=args.n_steps * args.n_envs,
                ent_coef=args.entropy_term,
                buffer_size=cfg.online_training.total_time_steps,
                learning_starts=cfg.online_training.learning_starts,
                seed=args.seed,
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
                learning_rate=args.learning_rate,
                train_freq=(
                    cfg.environment.train_freq_num,
                    cfg.environment.train_freq_type,
                ),
                gradient_steps=cfg.online_training.gradient_steps,
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)

    elif algo == "iql":
        model_class = IQL

        # policy = SACPolicy(observation_space=envs.observation_space, action_space=envs.action_space, net_arch=[32, 32], lr_schedule=None)
        if not args.pretrained:
            model = model_class(
                cfg.model.policy_type,
                envs,
                verbose=1,
                tensorboard_log=log_dir,
                buffer_size=cfg.online_training.total_time_steps,
                learning_starts=cfg.online_training.learning_starts,
                seed=args.seed,
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
                learning_rate=args.learning_rate,
                train_freq=(
                    cfg.environment.train_freq_num,
                    cfg.environment.train_freq_type,
                ),
                online_critic_update_ratio=cfg.online_training.critic_update_ratio,
                offline_critic_update_ratio=cfg.offline_training.critic_update_ratio,
                policy_extraction=cfg.general_training.policy_extraction,
                advantage_temp=cfg.general_training.awr_advantage_temp,
                ddpg_bc_weight=cfg.general_training.ddpg_bc_weight,
                n_critics_to_sample=cfg.general_training.n_critics_to_sample,
                warm_start_online_rl=cfg.online_training.warm_start_online_rl,
                gamma=cfg.general_training.gamma,
                action_chunk_size=cfg.general_training.action_chunk_size,
                gradient_steps=cfg.online_training.gradient_steps,
                success_bonus=cfg.reward_model.success_bonus,
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)

    if orig_algo.lower() == "rlpd":
        offline_model = model
        model_class = RLPD

        if not args.pretrained:
            model = model_class(
                cfg.model.policy_type,
                envs,
                offline_algo=None,
                verbose=1,
                tensorboard_log=log_dir,
                buffer_size=cfg.online_training.total_time_steps,
                learning_starts=cfg.online_training.learning_starts,
                seed=args.seed,
                action_noise=action_noise,  # should be null
                ent_coef=args.entropy_term,
                policy_kwargs=policy_kwargs,
                learning_rate=args.learning_rate,
                train_freq=(
                    cfg.environment.train_freq_num,
                    cfg.environment.train_freq_type,
                ),  # useless
                online_critic_update_ratio=cfg.online_training.critic_update_ratio,
                offline_critic_update_ratio=cfg.offline_training.critic_update_ratio,
                n_critics_to_sample=cfg.general_training.n_critics_to_sample,
                train_critic_with_entropy=cfg.general_training.rlpd_train_critic_with_entropy,
                warm_start_online_rl=cfg.online_training.warm_start_online_rl,
                gamma=cfg.general_training.gamma,
                action_chunk_size=cfg.general_training.action_chunk_size,
                success_bonus=cfg.reward_model.success_bonus,
                gradient_steps=cfg.online_training.gradient_steps,
            )
        else:
            model = model_class.load(args.pretrained, env=envs, tensorboard_log=log_dir)

    assert model is not None, "Model is None. Something went wrong."

    return model, model_class, policy_kwargs


def generate_callback_list(args: DictConfig, eval_callback: EvalCallback):
    if args.wandb:
        customwandbcallback = CustomWandbCallback()
        callbacks = [eval_callback, customwandbcallback]
        #callbacks = [customwandbcallback]
    else:
        # callbacks = [eval_callback]
        callbacks = []
    return callbacks


def custom_save_model(model, path):
    """
    Custom save function that excludes large components like reward models and replay buffers.

    Args:
        model: The model to save
        path: Path where to save the model
    """
    print(f"Custom saving model to {path}")

    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Save policy weights only
    if hasattr(model, "policy") and hasattr(model.policy, "state_dict"):
        policy_path = os.path.join(path, "policy.pth")
        th.save(model.policy.state_dict(), policy_path)
        print(f"Saved policy weights to {policy_path}")

    # Save actor weights if available
    if hasattr(model, "actor") and hasattr(model.actor, "state_dict"):
        actor_path = os.path.join(path, "actor.pth")
        th.save(model.actor.state_dict(), actor_path)
        print(f"Saved actor weights to {actor_path}")

    # Save critic weights if available
    if hasattr(model, "critic") and hasattr(model.critic, "state_dict"):
        critic_path = os.path.join(path, "critic.pth")
        th.save(model.critic.state_dict(), critic_path)
        print(f"Saved critic weights to {critic_path}")

    # Save value network weights for IQL
    if hasattr(model, "v_net") and hasattr(model.v_net, "state_dict"):
        v_net_path = os.path.join(path, "v_net.pth")
        th.save(model.v_net.state_dict(), v_net_path)
        print(f"Saved value network weights to {v_net_path}")

    # Save some basic parameters in a JSON file
    params = {
        "algorithm": model.__class__.__name__,
        "action_space": str(model.action_space),
        "observation_space": str(model.observation_space),
        "gamma": model.gamma if hasattr(model, "gamma") else None,
        "learning_rate": model.learning_rate
        if hasattr(model, "learning_rate")
        else None,
    }

    params_path = os.path.join(path, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

    print("Saved model parameters to", params_path)
    print("Custom model saving complete")


if __name__ == "__main__":
    main()
