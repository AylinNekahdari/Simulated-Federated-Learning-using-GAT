import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=2, heads=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)


def init_model(in_dim, device):
    return GATNet(in_dim, hidden_channels=32, out_channels=2).to(device)
