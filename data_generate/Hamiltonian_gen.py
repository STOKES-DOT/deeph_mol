from pyscf import gto, scf
import matplotlib.pyplot as plt
import numpy as np


class Hamiltonian_gen:
    def __init__(self,mol2):
        self.mol2=mol2
    def get_overlap(self,basis='sto-3g'):
        mol = gto.M(atom=self.xyz,basis=f'{basis}')
        S_ao = mol.intor('int1e_ovlp')
        return S_ao
    def get_hamiltonian(self,basis='sto-3g',xc='b3lyp'):
        mol = gto.M(atom=self.xyz,basis=f'{basis}')
        mf = scf.RKS(mol)
        mf.xc = xc
        fock = mf.get_fock()
        H_ao = fock + mol.get_veff()
        return H_ao
