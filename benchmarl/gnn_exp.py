import torch_geometric
from torch import nn

from environments.vmas.common import VmasTask
from algorithms.mappo import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import SequenceModelConfig, GnnConfig, MlpConfig


config=ExperimentConfig.get_from_yaml()
config.checkpoint_at_end = True
config.save_folder = "./checkpoints/gnn_exp_1_agents_3_adversaries_old/freeze_adversary_actor/"
# config.restore_file = "./checkpoints/gnn_exp_1_agents_3_adversaries_old/mappo_simple_tag_pos_gnn__c7b6724a_25_03_01-19_20_03/checkpoints/checkpoint_3000000.pt"
freeze_config = {
    "group": "adversary",
    "actor_critic": "actor",
    "model_path": "./checkpoints/gnn_exp_1_agents_3_adversaries_old/mappo_simple_tag_pos_gnn__c7b6724a_25_03_01-19_20_03/checkpoints/checkpoint_3000000.pt",
}
experiment = Experiment(
    algorithm_config=MappoConfig.get_from_yaml(),
    model_config=GnnConfig(
        topology="from_pos",
        self_loops=True,
        gnn_class=torch_geometric.nn.conv.GATv2Conv,
        position_key="pos",
        velocity_key="vel",
        exclude_pos_from_node_features=True,
        edge_radius=10,
        pos_features=2,
        vel_features=2,
        gnn_kwargs={},
    ),
    critic_model_config=MlpConfig(num_cells=[256,256], activation_class=nn.Tanh, layer_class=nn.Linear),
    seed=0,
    config=config,
    task=VmasTask.SIMPLE_TAG_POS.get_from_yaml(),
    freeze_config=freeze_config,
)
experiment.run()
# experiment._evaluation_loop_all_videos()