import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import edges_embedding
import nodes_embedding
import bond_embedding
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
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_of_heads, num_out_features, 1))
        self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_of_heads, num_out_features, 1))

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

    
class GATlayer(GATlayer_Base):
    def __init__(self, num_in_features, num_out_features, num_heads, dropout=0.1, concat=True,activation=nn.ELU(), add_skip_connection=True, bias=None, log_attention_weights=None):
        super(GATlayer, self).__init__(num_in_features, num_out_features, num_heads, concat=concat, activation=activation, add_skip_connection=add_skip_connection, bias=bias, log_attention_weights=log_attention_weights)
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.activation = activation
        self.add_skip_connection = add_skip_connection
        self.bias = bias
        self.log_attention_weights = log_attention_weights
    def forward(self,nodes_features,degree_matrix,edges_features_distance,edges_features_bond,cutoff=0):
        nodes = nodes_features
        edges1 = edges_features_distance
        edges2 = edges_features_bond
        weight_degree_matrix_net = degree_matrix+edges2
        connectivity_mask = torch.where(weight_degree_matrix_net>0,weight_degree_matrix_net,
            torch.where(
                edges2>cutoff,
                edges2+weight_degree_matrix_net,
                -1e9
            ))#edges with distance larger than cutoff will be masked
        num_of_nodes = nodes.size(0)
        assert connectivity_mask.shape == (num_of_nodes,num_of_nodes),f"connectivity_mask shape error,expected {(num_of_nodes,num_of_nodes)},got {connectivity_mask.shape}"
        
        nodes_features = self.dropout(nodes_features)
        nodes_features_proj = torch.matmul(nodes_features.unsqueeze(0),self.proj_param)
        nodes_features_proj = self.dropout(nodes_features_proj)

        scores_source = torch.bmm(nodes_features_proj,self.scoring_fn_source)
        scores_target = torch.bmm(nodes_features_proj,self.scoring_fn_target)
        all_scores = self.leakyReLU(scores_source + scores_target.transpose(1,2))
        all_attention_coefficients = torch.softmax(all_scores + connectivity_mask,dim=1)
        
        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj)
        out_nodes_features = out_nodes_features.transpose(0,1)
        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, nodes_features, out_nodes_features)
        connectivity_mask = nn.LayerNorm(connectivity_mask.shape[0])(connectivity_mask)
        return (out_nodes_features, connectivity_mask)
    
if __name__ == '__main__':
    mol2 = '/Users/jiaoyuan/Documents/GitHub/deeph_dft_molecules/deeph_mol/dataset/mol/3.mol2'
    nodes_embed = nodes_embedding.Nodes_Embedding(mol2)
    nodes_features = nodes_embed.forward()
    edges_embedding_embed = edges_embedding.Edges_Embedding(mol2)
    edges1, edges2, degree_tensor = edges_embedding_embed.forward()
    gat_layer = GATlayer(nodes_features.shape[1], nodes_features.shape[1], 3)
    out_nodes_features, connectivity_mask = gat_layer(nodes_features, degree_tensor, edges1, edges2)
    print(out_nodes_features)
    print(out_nodes_features.shape)
    print(connectivity_mask)
    print(connectivity_mask.shape)