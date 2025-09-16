from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

smiles = pd.read_csv('data_generate\smiles.csv')
smiles = smiles['smiles']

def mol_to_xyz(mol,index,files):
    Chem.MolToXYZFile(mol, f'{files}\{index}.xyz')
    print(f'{index}.xyz saved')
    
for i in range(len(smiles)):
    mol = Chem.MolFromMolFile(f'dataset\mol\{i+1}.mol')
    mol_to_xyz(mol,i+1,r'dataset\xyz')
