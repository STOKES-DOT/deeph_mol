import torch 
import torch.nn as nn

class Atom_Embedding(nn.Module):
    def __init__(self,num_atom_types,num_atom_types_chemistry,embedding_dim,embedding_dim_chemistry):
        super().__init__()
        self.embedding_dim_chemistry = embedding_dim_chemistry #Chemical environment embedding dimension
        self.embedding_dim = embedding_dim #Atom embedding dimension
        self.embedding = nn.Embedding(num_atom_types,embedding_dim)
        self.chemistry_embedding = nn.Embedding(num_atom_types_chemistry,embedding_dim_chemistry)
    