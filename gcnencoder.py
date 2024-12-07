import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import VGAE
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import APPNP
from torch_geometric.nn import RGCNConv, FastRGCNConv

from config import Config

config = Config()


# matrix normalization

def normalize_adj(mx):
    """
    Normalize adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize matrix

    Parameters
    ----------
    mx : torch.Tensor
        matrix to be normalized

    Returns
    -------
    torch.Tensor
        normalized matrix
    """

    if mx[0, 0] == 0:
        mx = mx + torch.eye(mx.shape[0], device=mx.device)
    rowsum = mx.sum(1)
    r_inv = torch.pow(rowsum, -1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    mx = torch.mm(mx, r_mat_inv)
    return mx


# VGAE ENCODER

class DenseGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseGCNConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        # self.weight1 = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.batch_norm = nn.BatchNorm1d(out_channels)
        # self.layer_norm = nn.LayerNorm(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        x = torch.mm(torch.mm(adj, x), self.weight)
        x = self.batch_norm(x)
        return x


class RGAE_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(RGAE_Encoder, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.skip_connection = None if not config.skip_layer() else nn.Linear(in_channels, out_channels)
        for i in range(config.num_cnn_layers()):
                conv = FastRGCNConv(in_channels, out_channels, num_relations)
                batch_norm = nn.BatchNorm1d(out_channels)
                self.conv_layers.append(conv)
                self.batch_norms.append(batch_norm)
                in_channels = out_channels
         
            

        self.act = config.activation_function()
        #print(self.act)
        self.act=nn.ELU()
        self.drop = nn.Dropout(config.drop_out())

        # Initialize weights using Xavier initialization
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, (nn.Linear, RGCNConv)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_types):
        skip_input = x

        for i, (conv_layer, batch_norm) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x = conv_layer(x, edge_index, edge_types)

            # Apply batch normalization to all but the last layer
            if config.batch_norm() and i < len(self.conv_layers) - 1:
                x = batch_norm(x)
            x = self.act(x)
            x = self.drop(x)
        
        if self.skip_connection is not None:
            skip_output = self.skip_connection(skip_input)
            x = x + skip_output
            x = self.act(x)
        
        return x


class RGAE_Encoder_old(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(RGAE_Encoder, self).__init__()

        # Initialize the first convolutional layer and batch normalization
        self.conv1 = FastRGCNConv(in_channels, out_channels, num_relations)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Initialize the second convolutional layer and batch normalization
        self.conv2 = FastRGCNConv(out_channels, out_channels, num_relations)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Initialize skip connection if required
        self.skip_connection = None if not config.skip_layer() else nn.Linear(in_channels, out_channels)

        # Activation function and dropout
        self.act = nn.ELU()
        self.drop = nn.Dropout(config.drop_out())

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Apply Xavier initialization to weights
        for m in self.modules():
            if isinstance(m, (nn.Linear, FastRGCNConv)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_types):
        # Save the input for the skip connection
        skip_input = x

        # First layer
        x = self.conv1(x, edge_index, edge_types)
        if config.batch_norm():  # Assuming 'config' is accessible and has a 'batch_norm' method
            x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)

        # Second layer
        x = self.conv2(x, edge_index, edge_types)
        x=self.act(x)
        x = self.drop(x)
        
        #x=torch.sigmoid(x)
        # Skip connection
        if self.skip_connection is not None:
            skip_output = self.skip_connection(skip_input)
            x = x + skip_output
        
            
        
        #x=torch.sigmoid(x)
        
        return x



class GAE_Encoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GAE_Encoder, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.skip_connection = None if not config.skip_layer() else nn.Linear(in_channels, out_channels)
        for i in range(config.num_cnn_layers()):
            conv = GCNConv(in_channels, out_channels)
            batch_norm = nn.BatchNorm1d(out_channels)
            self.conv_layers.append(conv)
            self.batch_norms.append(batch_norm)
            in_channels = out_channels

        self.act = config.activation_function()
        self.drop = nn.Dropout(config.drop_out())

        # Initialize weights using Xavier initialization
        self._init_weights()

    def forward(self, x, edge_index):
        skip_input = x

        for i, (conv_layer, batch_norm) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x = conv_layer(x, edge_index)

            # Apply batch normalization to all but the last layer
            if config.batch_norm() and i < len(self.conv_layers) - 1:
                x = batch_norm(x)
            x = self.act(x)
            x = self.drop(x)

        if self.skip_connection is not None:
            skip_output = self.skip_connection(skip_input)
            x = x + skip_output
            x = self.act(x)

        return x

    def _init_weights(self):
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
