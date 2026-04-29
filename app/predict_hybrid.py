import sys
import numpy as np
import joblib
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# =====================
# LOAD MODEL
# =====================
data = joblib.load("hybrid_model.pkl")

models = data["models"]
mlb = data["mlb"]
target_mlb = data["target_mlb"]
scaler = data["scaler"]
FP_SIZE = data["fp_size"]
TOP_K = data["top_k"]

# =====================
# INPUT
# =====================
smiles = sys.argv[1]

# =====================
# FEATURE BUILDING
# =====================
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_SIZE)
    fp_arr = np.array(fp, dtype=np.float32)

    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumAromaticRings(mol),
    ]

    return np.concatenate([fp_arr, desc])


# =====================
# FETCH TARGETS
# =====================
def get_targets(smiles):
    try:
        url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
        params = {
            "molecule_structures__canonical_smiles__connectivity": smiles
        }

        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        if not data["molecules"]:
            return []

        chembl_id = data["molecules"][0]["molecule_chembl_id"]

        r = requests.get(
            "https://www.ebi.ac.uk/chembl/api/data/activity.json",
            params={"molecule_chembl_id": chembl_id, "limit": 1000},
            timeout=10,
        )

        acts = r.json().get("activities", [])
        target_ids = list(set(a["target_chembl_id"] for a in acts if a.get("target_chembl_id")))

        names = []
        for tid in target_ids[:20]:
            r = requests.get(
                f"https://www.ebi.ac.uk/chembl/api/data/target/{tid}.json",
                timeout=10,
            )
            name = r.json().get("pref_name")
            if name:
                names.append(name)

        return list(set(names))

    except:
        return []


# =====================
# BUILD INPUT VECTOR
# =====================
x = smiles_to_features(smiles).reshape(1, -1)

# scale descriptors
x[:, FP_SIZE:] = scaler.transform(x[:, FP_SIZE:])

# targets
targets = get_targets(smiles)
T = target_mlb.transform([targets])

# concat
x = np.concatenate([x, T], axis=1)

# =====================
# PREDICT
# =====================
probs = []

for clf in models:
    if clf is None:
        probs.append(0.0)
    else:
        probs.append(clf.predict_proba(x)[0][1])

probs = np.array(probs)

# =====================
# TOP-K ONLY
# =====================
topk_idx = np.argsort(probs)[-TOP_K:][::-1]

print(f"\n--- Top-{TOP_K} predicted side effects ---\n")

for i in topk_idx:
    print(f"{mlb.classes_[i]:35s} | {probs[i]:.3f}")