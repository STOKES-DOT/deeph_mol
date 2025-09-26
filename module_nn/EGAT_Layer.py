import torch
import torch.nn as nn
import torch.nn.functional as F
import e3nn.o3 as o3
from e3nn.nn import Gate, FullyConnectedNet
import edges_embedding
import nodes_embedding
import bond_embedding
import math


class EGAT_layer_base(nn.Mudule):
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False, lmax=2, sh_dim=None):
        super().__init__()
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.num_heads = num_of_heads
        self.concat = concat
        self.lmax = lmax
        
        if sh_dim is None:
            self.sh_dim = (lmax + 1) ** 2
        else:
            self.sh_dim = sh_dim
        if add_skip_connection:
            self.skip_linear = nn.Linear(num_in_features, num_out_features, bias=False)
        else:
            self.skip_linear = None
    def init_params(self):
        # 使用E3NN推荐的参数初始化
        for tp in [self.tp_query, self.tp_key, self.tp_value, self.edge_tp_distance, self.edge_tp_bond]:
            for weight in tp.parameters():
                if weight.dim() > 1:
                    nn.init.xavier_uniform_(weight)
        
        if self.skip_linear is not None:
            nn.init.xavier_uniform_(self.skip_linear.weight)


class EGATlayer(EGAT_layer_base):
    def __init__(self, num_in_features, num_out_features, num_heads=1, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False, lmax=2, sh_dim=None):
        super(EGATlayer,self).init(
                num_in_features, 
                num_out_features, 
                num_heads=1, 
                concat=True, 
                activation=None, dropout_prob=0.6, add_skip_connection=True, 
                bias=True, log_attention_weights=False, lmax=2, sh_dim=None
        )
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.num_heads = num_heads
        self.concat = concat
        self.lmax = lmax
        self.dropout_prob = dropout_prob
        
        #nodes-scalars
        self.irreps_nodes_in = o3.Irrep(f"{num_in_features}x0e")
        self.irreps_nodes_out = o3.Irreps(f"{num_out_features}x0e")
        #nodes_vector-atom-pairs directions
        
        self.irreps_dir = o3.Irreps.spherical_harmonics(lmax=lmax)
        self.sh_calculator = o3.SphericalHarmonics(lmax, normalize=True, normalization='component')
        
        #equivalent K, Q, V
        self.tp_query = o3.FullyConnectedTensorProduct(
            self.irreps_nodes_in,
            self.irreps_dir,
            o3.Irreps(f"{num_heads}x0e")
        )
        self.tp_key = o3.FullyConnectedTensorProduct(
            self.irreps_nodes_in,
            self.irreps_dir,
            o3.Irreps(f"{num_heads}x0e")
        )
        
        self.tp_value = o3.FullyConnectedTensorProduct(
            self.irreps_nodes_out,
            self.irreps_dir,
            o3.Irreps(f"{num_out_features}x0e")
        )
        
        #edges equivariant embedding
        self.egde_tp_gaussian = o3.FullyConnectedTensorProduct(
            o3.Irreps('1x0e'),
            o3.Irreps('1x0e'),
            o3.Irreps('1x0e')
        )
        self.edge_tp_bond = o3.FullyConnectedTensorProduct(
            o3.Irreps('5x0e'),#five types of bonds
            o3.Irreps('1x0e'),
            o3.Irreps('1x0e')
        )
        
    def forward(self, node_features, edges_features_distance, edges_features_bond, edge_direction, degree_matrix, cut_off=5.0):
        num_nodes = node_features.shape[0]
        direction_vec = edge_direction
        
        sh_features = self.sh_calculator(direction_vec.reshape(-1,3))
        sh_features = sh_features.reshape(num_nodes, num_nodes, self.sh_dim)
        
        connectivity_mask = torch.where(degree_matrix>0, degree_matrix, -1e9)
        
        nodes_i = node_features.unsqueeze(1).expand(num_nodes, num_nodes, -1)
        nodes_j = node_features.unsqueeze(0).expand(num_nodes, num_nodes, -1)
        
        query_input = nodes_i.reshape(-1, self.num_in_features)
        sh_input = sh_features.reshape(-1,self.sh_dim)
        
        quert = self.tp_query(query_input, sh_input)
        quert = quert.reshape(num_nodes, num_nodes, self.num_heads)
        
        key_input = nodes_j.reshape(-1, self.num_in_features)
        sh_input = sh_features.reshape(-1,self.sh_dim)
        
        key = self.tp_key(key_input, sh_input)
        key = key.reshape(num_nodes, num_nodes, self.num_heads)
        
        raw_attention = (quert + key.transpose(0,1)) / math.sqrt(self.num_heads)
        
        edges_features_bond = self.edge_tp_bond(edges_features_bond, sh_features.reshape(-1,self.sh_dim))
        edge_bond_contribution = self.edge_tp_bond(
                edges_features_bond.reshape(-1, edges_features_bond.shape[-1]), 
                torch.ones(edges_features_bond.numel() // edges_features_bond.shape[-1], 1, device=edges_features_bond.device)
            ).reshape(num_nodes, num_nodes, 1)
        #bond type contribution to attention
        raw_attention += edge_bond_contribution
        
        edge_gaussian_contribution = self.edge_tp_gaussian(
            edges_features_distance.reshape(-1, 1), 
            torch.ones(edges_features_distance.numel(), 1, device=edges_features_distance.device)
        ).reshape(num_nodes, num_nodes, 1)
        #distance contribution to attention
        raw_attention += edge_gaussian_contribution
        
        attention_scores = self.leakyReLU(raw_attention)+connectivity_mask.unsqueeze(-1)
        
        attention_weight = F.softmax(attention_scores)
        attention_weight = F.dropout(attention_weight)
        
        sh_mean = sh_features.mean(dim=1)
        value_input = nodes_j.reshape(-1, self.num_in_features)
        sh_value_input = sh_mean.reshape(-1,self.sh_dim)
        value = self.tp_value(value_input, sh_value_input)
        value = value.reshape(num_nodes, num_nodes, self.num_out_features)
        
        attention_mean = attention_weight.mean(dim=2, keepdim=True)
        aggregated = torch.matmul(attention_mean.transpose, value)
        
        
        updated_nodes = self.activation(aggregated)
        
        with torch.no_grad():
            # 使用节点相似性和距离更新连通性
            node_similarity = torch.matmul(updated_nodes, updated_nodes.T)
            # 距离衰减因子
            distance_decay = -edges_features_distance.squeeze(-1)
            updated_connectivity = torch.sigmoid(node_similarity) * distance_decay * degree_matrix
            
            # 层归一化
            if updated_connectivity.numel() > 0:
                updated_connectivity = F.layer_norm(
                    updated_connectivity, 
                    updated_connectivity.shape[-1:]
                )
        
        return updated_nodes, updated_connectivity
        
        