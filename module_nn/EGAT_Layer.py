import torch
import torch.nn as nn
import torch.nn.functional as F
import e3nn.o3 as o3
from e3nn.nn import Gate, FullyConnectedNet
import edges_embedding
import nodes_embedding
import bond_embedding
import math
class GATlayer_Base(torch.nn.Module):
    head_dim = 1
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.proj_param = nn.Parameter(torch.Tensor(num_of_heads, num_in_features, num_out_features))
        self.scoring_fn_target = nn.Parameter(torch.Tensor(num_of_heads, num_out_features, 1))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(num_of_heads, num_out_features, 1))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2) 
        self.softmax = nn.Softmax(dim=-1)  
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  
        self.attention_weights = None 


    def init_params(self):

        nn.init.xavier_uniform_(self.proj_param)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  
            self.attention_weights = attention_coefficients
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()
        if out_nodes_features.dim() != 3 or out_nodes_features.size(1) != self.num_of_heads:
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads, self.num_out_features)
        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=1)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


    def init_params(self):

        nn.init.xavier_uniform_(self.proj_param)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  
            self.attention_weights = attention_coefficients
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()
        """AI is creating summary for skip_concat_bias

        Returns:
            [type]: [description]
        """
        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)
class tensor_field_net(GATlayer_Base):
    def __init__(self, num_in_features, num_out_features, num_heads=1, dropout=0.5, concat=True,activation=nn.ELU(), add_skip_connection=None, bias=None, log_attention_weights=True,lmax=2):
        super(tensor_field_net, self).__init__(num_in_features, num_out_features, num_heads, concat=concat, activation=activation, add_skip_connection=add_skip_connection, bias=bias, log_attention_weights=log_attention_weights)
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.lmax = lmax
        self.irreps_input = o3.Irreps.spherical_harmonics(lmax=self.lmax) * self.num_in_features
        self.irreps_output = o3.Irreps.spherical_harmonics(lmax=self.lmax) * self.num_out_features
        self.irreps_key = o3.Irreps.spherical_harmonics(lmax=self.lmax) * self.num_in_features
        self.irreps_query = o3.Irreps.spherical_harmonics(lmax=self.lmax) * self.num_out_features
        self.number_of_basis = self.irreps_input.dim
        self.shape = o3.Irreps.spherical_harmonics(lmax=self.lmax)
        self.tp = o3.FullyConnectedTensorProduct(self.irreps_input, self.shape, self.irreps_output)
        self.tp_k = o3.FullyConnectedTensorProduct(self.irreps_input, self.irreps_key, self.irreps_output)
        self.tp_q = o3.FullyConnectedTensorProduct(self.irreps_input, self.irreps_query, self.irreps_output)
        self.fc = FullyConnectedNet([self.number_of_basis, 2 * self.number_of_basis, self.number_of_basis], torch.relu)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()
    def forward(self,nodes_features,degree_matrix,edges_features_distance,edges_features_bond,nodes_features_vector=None,cutoff=0):
        dot = o3.FullyConnectedTensorProduct(self.irreps_input, self.irreps_input, o3.Irreps("0e"))
        q=self.tp_q(nodes_features, nodes_features)
        k=self.tp_k(nodes_features, nodes_features)
        v=self.tp(nodes_features, nodes_features)
        
        exp = (dot(q, k) / math.sqrt(self.number_of_basis)).exp()
        attention = exp / (exp.sum(dim=-1, keepdim=True) + 1e-10)
        attention = self.dropout(attention)
        out_nodes_features = attention * v
        return self.skip_concat_bias(attention, nodes_features, out_nodes_features)
    
    
    
if __name__ == "__main__":
    mol2 = '/Users/jiaoyuan/Documents/GitHub/deeph_dft_molecules/deeph_mol/dataset/mol/3.mol2'
    nodes_embed = nodes_embedding.Nodes_Embedding(mol2)
    nodes_features = nodes_embed.forward()
    edges_embedding_embed = edges_embedding.Edges_Embedding(mol2)
    edges1, edges2, degree_tensor = edges_embedding_embed.forward()
    test_tfn = tensor_field_net(nodes_features.shape[1], nodes_features.shape[1], 1).forward(nodes_features,degree_tensor,edges1,edges2,nodes_features,cutoff=0)
    print(test_tfn)
  
