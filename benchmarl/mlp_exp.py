import torch_geometric
from torch import nn

from benchmarl.environments.vmas.common import VmasTask
from benchmarl.algorithms.mappo import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import SequenceModelConfig, GnnConfig, MlpConfig


def configure_experiment():
    config = ExperimentConfig.get_from_yaml()
    config.checkpoint_at_end = True
    config.save_folder = f"./mlp_freeze_{freeze_config["group"]}/"  # setting save folder based on freeze_config

    # limit number of iterations for speed testing
    config.max_n_iters = 2  # comment out this line to do the full 500 iterations by default

    return config

def configure_task():
    task = VmasTask.SIMPLE_TAG.get_from_yaml()  # start with [vmas simple tag] config
    task.config['num_good_agents'] = 3
    task.config['num_adversaries'] = 3
    task.config['max_steps'] = 700  # increases number of evaluation steps (and corresponding length of video)

    # set 'shape_*_rew' to be True, meaning reward is based on increasing/decreasing distance
    task.config['shape_agent_rew'] = True
    task.config['shape_adversary_rew'] = True

    return task


freeze_config = {
    "group": "adversary",
    "actor_critic": "actor",
    "model_path": "./mlp_3v3/mappo_simple_tag_mlp__494cf3be_25_05_02-21_15_50/checkpoints/checkpoint_3000000.pt",
}


if __name__ == "__main__":
    experiment = Experiment(
        algorithm_config=MappoConfig.get_from_yaml(),
        model_config=MlpConfig(num_cells=[256,256], activation_class=nn.Tanh, layer_class=nn.Linear),
        critic_model_config=MlpConfig(num_cells=[256,256], activation_class=nn.Tanh, layer_class=nn.Linear),
        seed=0,
        config=configure_experiment(),
        task=configure_task(),
        freeze_config=freeze_config,
    )

    experiment.run()