import torch 
import torch.nn as nn
import numpy as np

class Atom_Embedding(nn.Module):
    def __init__(self,mol2):
        super().__init__()
        self.atom_type_dict = "deeph_mol/module_nn/atom_embedding_shell.dat"
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
    def atom_type_to_vector(self):  # atom type to shell conformation vector
        atom_types = self.atom_types
        with open(self.atom_type_dict, 'r') as f:
            lines = f.readlines()

        atom_vector_map = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(', [', 1)
            if len(parts) < 2:
                continue
            atom_symbol = parts[0].strip()
            vector_str = parts[1].rstrip(']')
            vector = [int(x) for x in vector_str.split(',')]
            atom_vector_map[atom_symbol] = vector

        result_vectors = []
        for atom in atom_types:
            if atom in atom_vector_map:
                result_vectors.append(atom_vector_map[atom])
            else:
                raise ValueError(f"Atom type {atom} not found in the dictionary")
    
        return result_vectors

  
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
    print(atom_embed.atom_type_to_vector())
    print(atom_embed.atom_molecular_part())