import torch 
import torch.nn as nn
import numpy as np

class Bond_Embedding(nn.Module):
    def __init__(self, mol2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mol2 = mol2
        self.bond_type_matrix = self.get_bond_type()
        self.gb_matrix = self.gaussian_basis_matrix()
    
    def get_atom_pairs_distance(self):
        atoms = []
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
                    atom_index = int(parts[0])
                    atom_name = parts[1][:2].title()
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    atoms.append((atom_index,atom_name,x,y,z))
        distance = []
        index = []
        n_atoms = len(atoms)
        for i in range(n_atoms):
            for j in range(i+1,n_atoms):
                index1, atom1, x1, y1, z1 = atoms[i]
                index2, atom2, x2, y2, z2 = atoms[j]
                distance.append(np.linalg.norm(np.array([x1,y1,z1])-np.array([x2,y2,z2]), ord=2))
                index.append((index1,index2))
        return distance, index
    
    def _gaussian_basis(self,rij,sigma=1,mu=0):
        if rij <= 10:
            rij=rij
        else:
            rij=0
        var_eps=1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(np.sqrt(rij)-mu)**2/(2*sigma**2))
        return var_eps
    def gaussian_basis_matrix(self,sigma,mu):
        distance, index = self.get_atom_pairs_distance()
        max_index = 0
        for idx_pair in index:
            max_index = max(max_index,max(idx_pair))
        gb_matrix = np.zeros((max_index, max_index))
        for dist, (i,j) in zip(distance,index):
            gb_value = self._gaussian_basis(dist,sigma,mu)
            gb_matrix[i-1,j-1] = gb_value
            gb_matrix[j-1,i-1] = gb_value
        return gb_matrix

    def get_bond_type(self):
        bond_type = []
        atom1 = []
        atom2 = []
        with open(self.mol2, 'r') as f:
            lines = f.readlines()
        atom_section = False
        for line in lines:
            if line.startswith('@<TRIPOS>BOND'):
                atom_section = True
                continue
            elif line.startswith('@<TRIPOS>'):
                atom_section = False
                continue
            if atom_section and line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    bond_type.append(parts[3])
                    atom1.append(int(parts[1]))
                    atom2.append(int(parts[2]))
        for i in range(len(atom1)):
            if bond_type[i] == '1':
                bond_type[i] = 1
            if bond_type[i] == '2':
                bond_type[i] = 2
            if bond_type[i] == '3':
                bond_type[i] = 3
            if bond_type[i] == 'ar':
                bond_type[i] = 1.5
            if bond_type[i] == 'am':
                bond_type[i] = 1.2
            else:
                bond_type[i] = 0
        max_index = max(max(atom1),max(atom2))
        bond_type_matrix = np.zeros((max_index, max_index))
        for values, (i,j) in zip(bond_type,zip(atom1,atom2)):
            bond_type_matrix[i-1,j-1] = values
            bond_type_matrix[j-1,i-1] = values
        return bond_type_matrix
    def save_matrix(self):
        from scipy.sparse import csr_matrix, save_npz
        gaussian_matrix = csr_matrix(self.gb_matrix)
        save_npz(f'{self.files}/{self.index}_g.npz',gaussian_matrix)
        bond_type_matrix = csr_matrix(self.bond_type_matrix)
        save_npz(f'{self.files}/{self.index}_b.npz',bond_type_matrix)
    
if __name__ == '__main__':
    bond_embedding = Bond_Embedding('/Users/jiaoyuan/Documents/GitHub/deeph_dft_molecules/deeph_mol/dataset/mol/3000.mol2')
    bond_embedding.get_bond_type()
    bond_type_matrix, bond_type = bond_embedding.get_bond_type()
    gb_matrix = bond_embedding.gaussian_basis_matrix(1,0)
    import matplotlib.pyplot as plt
    plt.imshow(bond_type_matrix)
    plt.savefig('bond_type_matrix.png')  # 保存当前显示的图像
    plt.show() 
    plt.imshow(gb_matrix)
    plt.savefig('gb_matrix.png')  # 保存当前显示的图像
    plt.show() 
