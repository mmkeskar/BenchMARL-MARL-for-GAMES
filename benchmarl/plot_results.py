# Relevant imports
from marl_eval.plotting_tools.plotting import (
    aggregate_scores,
    performance_profiles,
    probability_of_improvement,
    sample_efficiency_curves,
)
from marl_eval.utils.data_processing_utils import (
    create_matrices_for_rliable,
    data_process_pipeline,
)
import json

# Specify any metrics that should be normalised
METRICS_TO_NORMALIZE = ["return"]

# Read in and process data
with open("outputs/mappo_navigation_pos_gnn_working/mappo_navigation_pos_gnn_working.json", "r") as f:
    raw_data = json.load(f)

processed_data = data_process_pipeline(
    raw_data=raw_data, metrics_to_normalize=METRICS_TO_NORMALIZE
)

environment_comparison_matrix, sample_effeciency_matrix = create_matrices_for_rliable(
    data_dictionary=processed_data,
    environment_name="env_1",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)

# Generate performance profile plot
fig = performance_profiles(
    environment_comparison_matrix,
    metric_name="return",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)