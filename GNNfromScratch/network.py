import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_qubit, num_gate_types, num_feature_idx):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

        self.q_head = nn.Linear(hidden_dim, num_qubit)
        self.g_head = nn.Linear(hidden_dim, num_gate_types)
        self.p_head = nn.Linear(hidden_dim, num_feature_idx)
        self.t_head = nn.Linear(hidden_dim, num_qubit - 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = x.mean(dim=0)  # global pooling
        x = F.relu(self.linear(x))
        return self.q_head(x), self.g_head(x), self.p_head(x), self.t_head(x)
