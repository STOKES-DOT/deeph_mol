import torch
import torch.nn as nn
import torch.nn.functional as F
import e3nn.o3 as o3
from e3nn.nn import Gate, FullyConnectedNet
import edges_embedding
import nodes_embedding
import bond_embedding
import math


class EGAT_layer_base(nn.Module):
    def __init__(self, num_in_features, num_out_features, num_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False, lmax=4):
        super().__init__()
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.num_heads = num_heads
        self.concat = concat
        self.lmax = lmax
        
        if add_skip_connection:
            self.skip_linear = nn.Linear(num_in_features, num_out_features, bias=False)
        else:
            self.skip_linear = None
    def init_params(self):
        for tp in [self.tp_query, self.tp_key, self.tp_value, self.edge_tp_distance, self.edge_tp_bond]:
            for weight in tp.parameters():
                if weight.dim() > 1:
                    torch.manual_seed(42)
                    nn.init.xavier_uniform_(weight)
        
        if self.skip_linear is not None:
            nn.init.xavier_uniform_(self.skip_linear.weight)


class EGATlayer(EGAT_layer_base):
    def __init__(self, num_in_features, num_out_features, num_heads=1, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False, lmax=4):
        super(EGATlayer,self).__init__(
                num_in_features, 
                num_out_features, 
                num_heads=1, 
                concat=True, 
                activation=None, dropout_prob=0.6, add_skip_connection=True, 
                bias=True, log_attention_weights=False, lmax=4
        )
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.num_heads = num_heads
        self.concat = concat
        self.lmax = lmax
        self.sh_dim = (lmax+1)**2
        self.dropout_prob = dropout_prob
        
        #nodes-scalars
        self.irreps_nodes_in = o3.Irreps(f"{num_in_features}x0e")
        self.irreps_nodes_out = o3.Irreps(f"{num_out_features}x0e")
        #nodes_vector-atom-pairs directions
        
        self.irreps_dir = o3.Irreps.spherical_harmonics(lmax=lmax)
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
            self.irreps_nodes_in,
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
            o3.Irreps('1x0e'),
            self.irreps_dir,
            o3.Irreps('1x0e')
        )
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation
    def forward(self, node_features, edges_features_distance, edges_features_bond, edge_direction, degree_matrix, cut_off=5.0):
        num_nodes = node_features.shape[0]
        num_edges = edge_direction.shape[0]
        sh_features = o3.spherical_harmonics(self.irreps_dir, edge_direction, normalize=True, normalization='component')
        sh_features = sh_features.reshape(num_nodes,num_nodes, (self.lmax+1)**2)
        #mask for disconnected nodes
        connectivity_mask = torch.where(degree_matrix>0, 0.0, -1e9)
        
        nodes_i = node_features.unsqueeze(1).expand(num_nodes, num_nodes, -1)
        nodes_j = node_features.unsqueeze(0).expand(num_nodes, num_nodes, -1)
        
        
        query_input = nodes_i.reshape(num_nodes*num_nodes, self.num_in_features)
        sh_input = sh_features.reshape(-1,(self.lmax+1)**2)    
        query = self.tp_query(query_input, sh_input)
        query = query.reshape(num_nodes, num_nodes, self.num_heads)
        
        key_input = nodes_j.reshape(-1, self.num_in_features)
        key = self.tp_key(key_input, sh_input)
        key = key.reshape(num_nodes, num_nodes, self.num_heads)
        
        raw_attention = (query + key.transpose(0,1)) / math.sqrt(self.num_heads)

        
        
        #bond type embedding
        edges_features_bond = edges_features_bond.unsqueeze(-1).reshape(-1, 1)        
        edges_features_bond = self.edge_tp_bond(edges_features_bond, sh_features.reshape(-1,self.sh_dim))
        
        self.edge_tp_bond_contribution = o3.FullyConnectedTensorProduct(
            o3.Irreps(f"{edges_features_bond.shape[-1]}x0e"),
            o3.Irreps('1x0e'),
            o3.Irreps('1x0e')
        )
        
        edge_bond_contribution = self.edge_tp_bond_contribution(
                edges_features_bond.reshape(-1, edges_features_bond.shape[-1]), 
                torch.ones(edges_features_bond.numel() // edges_features_bond.shape[-1], 1, device=edges_features_bond.device)
            ).reshape(num_nodes, num_nodes, 1)
        #bond type contribution to attention
        raw_attention += edge_bond_contribution

        
        
        edge_gaussian_contribution = self.egde_tp_gaussian(
            edges_features_distance.reshape(-1, 1), 
            torch.ones(edges_features_distance.numel(), 1, device=edges_features_distance.device)
        ).reshape(num_nodes, num_nodes, 1)
        #distance contribution to attention
        
        
        raw_attention += edge_gaussian_contribution
        
        attention_scores = self.leakyReLU(raw_attention)+connectivity_mask.unsqueeze(-1)
        
        attention_weight = F.softmax(attention_scores)
        attention_weight = F.dropout(attention_weight)
        
        value_input = nodes_j.reshape(num_nodes*num_nodes, self.num_in_features)
        sh_value_input = sh_features.reshape(-1,self.sh_dim)
    

        value = self.tp_value(value_input, sh_value_input)
        value = value.reshape(num_nodes, num_nodes, self.num_out_features)
        attention_mean = attention_weight.mean(dim=2, keepdim=True)
        
        
        aggregated = torch.matmul(torch.Tensor(attention_mean.transpose(2,1)), value)
        updated_nodes = self.activation(aggregated)

        with torch.no_grad():
            updated_nodes = updated_nodes.reshape(num_nodes, self.num_out_features)
            node_similarity = torch.matmul(updated_nodes, updated_nodes.T)
            distance_decay = -edges_features_distance.squeeze(-1)
            updated_connectivity = torch.sigmoid(node_similarity) * distance_decay * degree_matrix
            
            if updated_connectivity.numel() > 0:
                updated_connectivity = F.layer_norm(
                    updated_connectivity, 
                    updated_connectivity.shape[-1:]
                )
            updated_connectivity = updated_connectivity + updated_connectivity.T
        return updated_nodes, updated_connectivity
        
if __name__ == '__main__':
    mol2 = '/Users/jiaoyuan/Documents/GitHub/deeph_dft_molecules/deeph_mol/dataset/mol/1.mol2'
    nodes_embed = nodes_embedding.Nodes_Embedding(mol2)
    nodes_features = nodes_embed.forward()
    edges_embedding_embed = edges_embedding.Edges_Embedding(mol2)
    edges1, edges2, degree_tensor = edges_embedding_embed.forward()
    egat_layer = EGATlayer(nodes_features.shape[1], nodes_features.shape[1], 1)
    egde_vec = bond_embedding.Bond_Embedding(mol2).get_atom_pairs_direction()
    edges_direction = torch.tensor(egde_vec, dtype=torch.float32)
    out_nodes_features, connectivity_mask = egat_layer.forward(nodes_features, edges1, edges2, edges_direction ,degree_tensor, cut_off=5.0)

    print("connectivity_mask:", connectivity_mask)
    print("out_nodes_features:", out_nodes_features)
    