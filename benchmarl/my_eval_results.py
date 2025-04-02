import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_theme()


# directories = {
#     "full_top": "outputs/mappo_simple_tag_pos_gnn_full_top",
#     "from_pos_r1": "outputs/mappo_simple_tag_pos_gnn_from_pos",
#     "from_pos_r10": "outputs/mappo_simple_tag_pos_gnn_from_pos_r10"
# }
# directories = {
#     "from_pos_r10": "checkpoints/gnn_exp_3_agents_3_adversaries/mappo_simple_tag_pos_gnn_from_pos_r10"
# }

graph_files = {
    "adversary": "collection_adversary_reward_episode_reward_mean.csv",
    "agent": "collection_agent_reward_episode_reward_mean.csv",
}

directories = {
    # "3 agents 3 adversaries big": "checkpoints/gnn_exp_3_agents_3_adv_big_model/mappo_simple_tag_pos_gnn__09cc955e_25_03_02-12_58_21",
    "1 agent 3 adversaries": "checkpoints/gnn_exp_1_agents_3_adversaries_old/freeze_adversary_actor",
    # "2 agents 2 adversaries": "checkpoints/gnn_exp_2_agents_2_adversaries/mappo_simple_tag_pos_gnn__68f3be5b_25_03_02-14_39_12",
    # "3 agents 3 adversaries big model more exploration": "checkpoints/gnn_exp_3_agents_3_adversaries_more_exp_big_model/mappo_simple_tag_pos_gnn__341f8af3_25_03_05-08_35_17"
}

# graph_files = {
#     "adversary critic": "train_adversary_loss_critic.csv",
#     "adversary policy": "train_adversary_loss_objective.csv",
#     "agent critic": "train_agent_loss_critic.csv",
#     "agent policy": "train_agent_loss_objective.csv"
# }

fig = plt.figure(figsize=(10, 10))
for title, directory in directories.items():
    direc = os.path.join(directory, directory.split("/")[2], "scalars")

    for agent, file_name in graph_files.items():
        data = pd.read_csv(os.path.join(direc, file_name), names=["timesteps", "reward"], header=None)
        plt.plot(data["timesteps"], data["reward"], label=" ".join([title, agent]))

plt.legend()
plt.show()
