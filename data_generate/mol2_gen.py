import pandas as pd
import pybel
smiles = pd.read_csv('/Users/jiaoyuan/Documents/GitHub/deeph_dft_molecules/data_generate/smiles.csv')
smiles = smiles['smiles'].tolist()


def smi_to_mol2(smiles,index,files):
    mol = pybel.readstring('smi', smiles)
    mol.addh()
    mol.make3D()
    mol.write('mol2', f'{files}/{index}.mol2', overwrite=True)
    print(f'{index}.mol2 saved')
    
for smi in enumerate(smiles):
    smi_to_mol2(smi[1],smi[0],'dataste/mol2')
