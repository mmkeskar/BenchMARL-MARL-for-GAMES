#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING

import torch
from vmas import render_interactively
from vmas.simulator.core import Agent


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    num_good_agents: int = MISSING
    num_adversaries: int = MISSING
    num_landmarks: int = MISSING
    shape_agent_rew: bool = MISSING
    shape_adversary_rew: bool = MISSING
    agents_share_rew: bool = MISSING
    adversaries_share_rew: bool = MISSING
    observe_same_team: bool = MISSING
    observe_pos: bool = MISSING
    observe_vel: bool = MISSING
    bound: float = MISSING
    respawn_at_catch: bool = MISSING

from vmas.scenarios.mpe.simple_tag import Scenario

def observation(self, agent: Agent):
    # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)

        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            if agent.adversary and not other.adversary:
                other_pos.append(other.state.pos - agent.state.pos)
                other_vel.append(other.state.vel)
            elif not agent.adversary and not other.adversary and self.observe_same_team:
                other_pos.append(other.state.pos - agent.state.pos)
                other_vel.append(other.state.vel)
            elif not agent.adversary and other.adversary:
                other_pos.append(other.state.pos - agent.state.pos)
            elif agent.adversary and other.adversary and self.observe_same_team:
                other_pos.append(other.state.pos - agent.state.pos)

        return {
            "obs": torch.cat(
            [
                *entity_pos,
                *other_pos,
                *other_vel,
            ],
            dim=-1,
            ),
            "pos": agent.state.pos,
            "vel": agent.state.vel
        }

# def observation(self, agent: Agent):
#     # get positions of all entities in this agent's reference frame
#     entity_pos = []
#     for entity in self.world.landmarks:
#         entity_pos.append(entity.state.pos - agent.state.pos)

#     # Create a magic number for now
#     # TODO: figure out how to customize this magic number based on the loaded model
#     pos_k = 3 # Total number of agents - 1
#     vel_k = 1 # Total number of good agents

#     if not agent.adversary:
#         vel_k -= 1

#     other_pos = []
#     other_vel = []
#     agent_pos = []
#     for other in self.world.agents:
#         if other is agent:
#             continue
#         if agent.adversary and not other.adversary:
#             other_pos.append(other.state.pos - agent.state.pos)
#             other_vel.append(other.state.vel)
#             agent_pos.append(other.state.pos - agent.state.pos)
#         elif not agent.adversary and not other.adversary and self.observe_same_team:
#             other_pos.append(other.state.pos - agent.state.pos)
#             other_vel.append(other.state.vel)
#             agent_pos.append(other.state.pos - agent.state.pos)
#         elif not agent.adversary and other.adversary:
#             other_pos.append(other.state.pos - agent.state.pos)
#         elif agent.adversary and other.adversary and self.observe_same_team:
#             other_pos.append(other.state.pos - agent.state.pos)

#     positions = torch.permute(torch.stack(other_pos), (1, 0, 2))
#     distances = torch.linalg.norm(torch.permute(torch.stack(other_pos), (1, 0, 2)), dim=2)
    
#     if len(other_pos) > pos_k:
#         _, topk_pos_indices = torch.topk(-distances, k=pos_k, dim=1)
#         ordered_topk_pos_indices = torch.argsort(topk_pos_indices, dim=1)
#         topk_pos_indices = torch.gather(topk_pos_indices, 1, ordered_topk_pos_indices).unsqueeze(-1).expand(-1, -1, 2)
#         positions = torch.gather(positions, 1, topk_pos_indices)
#     elif len(other_pos) < pos_k:
#         zeros_3d = torch.zeros(positions.shape[0], pos_k - len(other_pos), 2)
#         positions = torch.cat((positions, zeros_3d), dim=1)

#     other_pos = torch.unbind(torch.permute(positions, (1, 0, 2)), dim=0)
#     if vel_k > 0:
#         if len(other_vel) > 0:
#             velocities = torch.permute(torch.stack(other_vel), (1, 0, 2))
#             agent_dists = torch.linalg.norm(torch.permute(torch.stack(agent_pos), (1, 0, 2)), dim=2)
#             if len(other_vel) > vel_k:
#                 _, topk_vel_indices = torch.topk(-agent_dists, k=vel_k, dim=1)
#                 ordered_topk_vel_indices = torch.argsort(topk_vel_indices, dim=1)
#                 topk_vel_indices = torch.gather(topk_vel_indices, 1, ordered_topk_vel_indices).unsqueeze(-1).expand(-1, -1, 2)
#                 velocities = torch.gather(velocities, 1, topk_vel_indices)
#             elif len(other_vel) < vel_k:
#                 zeros_3d = torch.zeros(velocities.shape[0], vel_k - len(other_vel), 2)
#                 velocities = torch.cat((velocities, zeros_3d), dim=1)
#             other_vel = torch.unbind(torch.permute(velocities, (1, 0, 2)), dim=0)
#         else:
#             zeros_3d = torch.zeros(positions.shape[0], vel_k, 2)
#             other_vel = torch.unbind(torch.permute(zeros_3d, (1, 0, 2)), dim=0)
#     else:
#         other_vel = []
#     return {
#         "obs": torch.cat(
#         [
#             *entity_pos,
#             *other_pos,
#             *other_vel,
#         ],
#         dim=-1,
#         ),
#         "pos": agent.state.pos,
#         "vel": agent.state.vel
#     }


Scenario.observation = observation
SimpleTagScenario = Scenario


if __name__ == "__main__":
    render_interactively(
        SimpleTagScenario(),
        control_two_agents=True,
    )