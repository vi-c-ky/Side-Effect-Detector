from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import shap
import io
import base64
import traceback
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

print("Loading model...")
data = joblib.load("hybrid_model.pkl")
models = data["models"]
mlb = data["mlb"]
scaler = data.get("scaler")
fp_size = data["fp_size"]
print(f"Model loaded. fp_size={fp_size}, classes={len(mlb.classes_)}")

# ── Feature pipeline ─────────────────────────────────────────────────────────
# Multi-radius fingerprints (r2, r3, r4, r5) @ 1024 bits each = 4096 bits
# + 14 RDKit physicochemical descriptors
RADII = [2, 3, 4, 5]
BITS_PER_RADIUS = 1024
TOTAL_FP_BITS = BITS_PER_RADIUS * len(RADII)  # 4096

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Multi-radius fingerprints
    fps = []
    for r in RADII:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=BITS_PER_RADIUS)
        fps.append(np.array(fp, dtype=np.float32))
    fp_array = np.concatenate(fps)

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
        Descriptors.MinPartialCharge(mol),
    ], dtype=np.float32)

    if scaler is not None:
        desc = scaler.transform(desc.reshape(1, -1))[0]

    x = np.concatenate([fp_array, desc]).reshape(1, -1)
    return x, mol


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return send_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        smiles = request.json.get("smiles", "").strip()
        x, mol = smiles_to_features(smiles)

        if x is None:
            return jsonify({"error": "Invalid SMILES string"})

        probs = np.array([clf.predict_proba(x)[0][1] for clf in models])
        top_idx = np.argsort(probs)[-15:][::-1]

        results = [
            {"effect": mlb.classes_[i], "score": float(probs[i])}
            for i in top_idx
        ]
        return jsonify({"results": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


def extract_shap_values(explainer, x):
    """
    Handle SHAP output across library versions.
    Returns a 1-D array of SHAP values for the positive class.
    """
    sv = explainer.shap_values(x)

    # New SHAP (>=0.40): returns Explanation object
    if hasattr(sv, "values"):
        vals = sv.values
        # shape may be (1, n_features) or (1, n_features, 2)
        if vals.ndim == 3:
            return vals[0, :, 1]   # class 1
        return vals[0]

    # Old SHAP: returns list [class0_array, class1_array]
    if isinstance(sv, list):
        arr = sv[1] if len(sv) > 1 else sv[0]
        return arr[0] if arr.ndim == 2 else arr

    # Fallback: plain ndarray
    return sv[0] if sv.ndim == 2 else sv


def build_atom_weights(mol, shap_vals):
    """
    Map fingerprint SHAP values back to atoms using bitInfo from all radii.
    Contributions are spread evenly across atoms in each environment.
    """
    num_atoms = mol.GetNumAtoms()
    atom_weights = np.zeros(num_atoms)

    offset = 0
    for r in RADII:
        bi = {}
        AllChem.GetMorganFingerprintAsBitVect(
            mol, r, nBits=BITS_PER_RADIUS, bitInfo=bi
        )
        for bit_idx, environments in bi.items():
            global_bit = offset + bit_idx
            if global_bit >= len(shap_vals):
                continue
            sv = shap_vals[global_bit]
            for atom_idx, radius in environments:
                env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                atoms_in_env = {atom_idx}
                for bond_idx in env_bonds:
                    bond = mol.GetBondWithIdx(bond_idx)
                    atoms_in_env.add(bond.GetBeginAtomIdx())
                    atoms_in_env.add(bond.GetEndAtomIdx())
                for a in atoms_in_env:
                    atom_weights[a] += sv / len(atoms_in_env)
        offset += BITS_PER_RADIUS

    return atom_weights


@app.route("/shap", methods=["POST"])
def shap_view():
    try:
        smiles = request.json.get("smiles", "").strip()
        effect = request.json.get("effect", "").strip()

        if not smiles or not effect:
            return jsonify({"error": "Missing smiles or effect"})

        # Find class index
        classes = list(mlb.classes_)
        if effect not in classes:
            return jsonify({"error": f"Unknown effect: {effect}"})
        effect_idx = classes.index(effect)

        x, mol = smiles_to_features(smiles)
        if x is None or mol is None:
            return jsonify({"error": "Invalid SMILES string"})

        # Compute SHAP
        clf = models[effect_idx]
        explainer = shap.TreeExplainer(clf)
        shap_vals = extract_shap_values(explainer, x)

        print(f"[SHAP] effect={effect} idx={effect_idx} shap shape={shap_vals.shape} x shape={x.shape}")

        # Map to atoms
        atom_weights = build_atom_weights(mol, shap_vals)

        # Normalise so the map uses the full colour range
        max_abs = np.abs(atom_weights).max()
        if max_abs > 0:
            atom_weights = atom_weights / max_abs

        # Render similarity map using RDKit drawer (newer API)
        from rdkit.Chem.Draw import rdMolDraw2D
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol,
            atom_weights.tolist(),
            draw2d=drawer,
            colorMap="RdBu_r",
        )
        drawer.FinishDrawing()
        buf = io.BytesIO(drawer.GetDrawingText())
        buf.seek(0)

        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        return jsonify({"image": img_b64, "effect": effect})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)