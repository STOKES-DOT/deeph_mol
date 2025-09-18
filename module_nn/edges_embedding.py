import numpy as np
import torch
import torch.nn as nn
import bond_embedding

class Edges_Embedding(nn.Module):#edge feature embedding with MLP
    def __init__(self,mol2):
        super().__init__()
        self.mol2 = mol2
        self.bond_embed = bond_embedding.Bond_Embedding(self.mol2)
        gb_matrix = self.bond_embed.gaussian_basis_matrix()
        bond_type_matrix = self.bond_embed.get_bond_type()
        self.bond_matrix = bond_type_matrix
        self.register_buffer('gb_matrix', torch.tensor(gb_matrix, dtype=torch.float32))
        self.register_buffer('bond_type_matrix', torch.tensor(bond_type_matrix, dtype=torch.float32))

        self.edges_size = self.gb_matrix.shape[1]
        
        self.features_dim = self.gb_matrix.shape[1]
        self.flatten = nn.Flatten()
        self.edges_embedding = nn.Sequential(
            nn.Linear(self.features_dim,self.features_dim),
            nn.Sigmoid(),
            nn.Linear(self.features_dim,self.features_dim),
            nn.ReLU(),
            nn.Linear(self.features_dim,self.features_dim),
            nn.Sigmoid(),
            nn.Linear(self.features_dim,self.features_dim),
            nn.LeakyReLU(),
        )
    def get_degree_matrix(self):
        self.num_atoms = self.bond_type_matrix.shape[0]
        degree_matrix = np.zeros((self.num_atoms,self.num_atoms))
        for i in range(self.num_atoms):
            for j in range(self.num_atoms):
                degree_matrix[i,j] = 0
            if self.bond_matrix[i,j] != 0:
                degree_matrix[i,j] = 1
        return degree_matrix
    def forward(self):
        edge_info1 = self.edges_embedding(self.gb_matrix)
        edge_info2 = self.edges_embedding(self.bond_type_matrix)
        degree_matrix = self.get_degree_matrix()
        return edge_info1, edge_info2, degree_matrix
    
    
if __name__ == '__main__':
    mol2 = '/Users/jiaoyuan/Documents/GitHub/deeph_dft_molecules/deeph_mol/dataset/mol/2.mol2'
    edges_embed = Edges_Embedding(mol2)
    nodes1, nodes2, degree_matrix = edges_embed.forward()
    print(degree_matrix.max())