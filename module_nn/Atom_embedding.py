import torch 
import torch.nn as nn
import numpy as np

class Atom_Embedding(nn.Module):
    def __init__(self,mol2):
        super().__init__()
        self.mol2 = mol2
        self.atom_types = self.get_atom_type()
    def get_atom_type(self):
        atom_types = []
        with open(self.mol2, 'r') as f:
            lines = f.readlines()
        atom_section = False
        for line in lines:
            if line.startswith('@<TRIPOS>ATOM'):
                atom_section = True
                continue
            elif line.startswith('@<TRIPOS>'):
                atom_section = False
                continue
            if atom_section and line.strip():
                parts = line.split()
                if len(parts) >= 6: 
                    atom_symbol = parts[1][:2].title()
                    atom_types.append(atom_symbol)
        return atom_types
    def atom_type_to_vector(self):#atom type to shell conformation vector
        atom_types = self.atom_types
        atom_embedding_vectors_dict = np.zeros((len(atom_types),len(atom_types)))
        for atom in atom_types:
            if atom == 'H':
                atom_embedding_vectors_dict[atom_types.index(atom)] = [1,0,0,0,0,0,0,0,0,0]
            elif atom == 'C':
                atom_embedding_vectors_dict[atom_types.index(atom)] = [0,1,0,0,0,0,0,0,0,0]
            elif atom == 'N':
                atom_embedding_vectors_dict[atom_types.index(atom)] = [0,0,1,0,0,0,0,0,0,0]
            elif atom == 'O':
                atom_embedding_vectors_dict[atom_types.index(atom)] = [0,0,0,1,0,0,0,0,0,0]
            elif atom == 'F':
                atom_embedding_vectors_dict[atom_types.index(atom)] = [0,0,0,0,1,0,0,0,0,0]
            elif atom == 'P':
                atom_embedding_vectors_dict[atom_types.index(atom)] = [0,0,0,0,0,1,0,0,0,0]
            elif atom == 'S':
                atom_embedding_vectors_dict[atom_types.index(atom)] = [0,0,0,0,0,0,1,0,0,0]
            elif atom == 'Cl':
                atom_embedding_vectors_dict[atom_types.index(atom)] = [0,0,0,0,0,0,0,1,0,0]
            elif atom == 'Br':
                atom_embedding_vectors_dict[atom_types.index(atom)] = [0,0,0,0,0,0,0,1/2**(1/2),1/2**(1/2),0]
            elif atom == 'I':
                atom_embedding_vectors_dict[atom_types.index(atom)] = [0,0,0,0,0,0,0,0,1/2**(1/2),1/2**(1/2)]

        return atom_embedding_vectors_dict
    def atom_molecular_part(self):
        atom_molecule_part = []
        with open(self.mol2, 'r') as f:
            lines = f.readlines()
            atom_section = False
        with open(self.mol2, 'r') as f:
            lines = f.readlines()
        atom_section = False
        for line in lines:
            if line.startswith('@<TRIPOS>ATOM'):
                atom_section = True
                continue
            elif line.startswith('@<TRIPOS>'):
                atom_section = False
                continue
            if atom_section and line.strip():
                parts = line.split()
                if len(parts) >= 6: 
                    atom_symbol = parts[6][:2].title()
                    atom_molecule_part.append(int(atom_symbol))
        return atom_molecule_part

if __name__ == '__main__':
    atom_embed = Atom_Embedding('/Users/jiaoyuan/Documents/GitHub/deeph_dft_molecules/deeph_mol/dataset/mol/1.mol2')
    print(atom_embed.atom_types)
    print(atom_embed.atom_molecular_part())