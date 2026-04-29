import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# Load once
data = joblib.load("hybrid_model.pkl")

models = data["models"]
mlb = data["mlb"]
scaler = data["scaler"]
fp_size = data["fp_size"]

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Fingerprints (multi-radius)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
    fp3 = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
    fp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 5, nBits=1024)

    fp = np.concatenate([fp1, fp2, fp3, fp4])

    # Descriptors
    desc = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.NOCount(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol)
    ])

    # Scale descriptors
    desc = scaler.transform(desc.reshape(1, -1))[0]

    return np.concatenate([fp, desc])


def predict(smiles, top_k=15):
    x = smiles_to_features(smiles)
    if x is None:
        return None

    x = x.reshape(1, -1)

    probs = []
    for clf in models:
        probs.append(clf.predict_proba(x)[0][1])

    probs = np.array(probs)

    top_idx = np.argsort(probs)[-top_k:][::-1]

    results = [
        (mlb.classes_[i], float(probs[i]))
        for i in top_idx
    ]

    return results