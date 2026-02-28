"""
=============================================================
  PHASE 1 — Train & Save Oral Cancer SVM Model
=============================================================
  Usage:
      python phase1_train.py --zip path/to/your_dataset.zip

  Output:
      oral_cancer_svm_model.pkl  ← trained SVM model
      oral_cancer_scaler.pkl     ← feature scaler
=============================================================
"""

import os
import sys
import argparse
import zipfile
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────
IMG_SIZE             = (128, 128)
HOG_ORIENTATIONS     = 9
HOG_PIXELS_PER_CELL  = (8, 8)
HOG_CELLS_PER_BLOCK  = (2, 2)
MODEL_SAVE_PATH      = "oral_cancer_svm_model.pkl"
SCALER_SAVE_PATH     = "oral_cancer_scaler.pkl"
CANCER_FOLDER        = "cancer"
NON_CANCER_FOLDER    = "non_cancer"
SUPPORTED_EXTS       = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


# ──────────────────────────────────────────────
#  FEATURE EXTRACTION
# ──────────────────────────────────────────────
def extract_features(image_path):
    try:
        img  = Image.open(image_path).convert("L")
        gray = np.array(img.resize(IMG_SIZE), dtype=np.float32) / 255.0
        features = hog(
            gray,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            visualize=False
        )
        return features
    except Exception as e:
        print(f"  [SKIP] {image_path}: {e}")
        return None


# ──────────────────────────────────────────────
#  DATASET LOADER
# ──────────────────────────────────────────────
def load_dataset(data_dir):
    X, y = [], []
    for label_name, label_val in [(CANCER_FOLDER, 1), (NON_CANCER_FOLDER, 0)]:
        folder = os.path.join(data_dir, label_name)
        if not os.path.exists(folder):
            print(f"\n[ERROR] Folder not found: '{label_name}' inside the zip.")
            print(f"        Make sure your zip has exactly: cancer/ and non_cancer/ folders.")
            sys.exit(1)

        files = [f for f in os.listdir(folder) if f.lower().endswith(SUPPORTED_EXTS)]
        print(f"  → '{label_name}/' : {len(files)} images found")

        for fname in files:
            feats = extract_features(os.path.join(folder, fname))
            if feats is not None:
                X.append(feats)
                y.append(label_val)

    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
#  MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────
def train(zip_path):
    print("\n" + "="*55)
    print("   PHASE 1 — Oral Cancer SVM Training")
    print("="*55)

    # Step 1: Extract zip
    print("\n[1/5] Extracting zip file ...")
    extract_dir = "oral_cancer_dataset"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"      Extracted to: {extract_dir}/")

    # Handle single nested folder inside zip
    sub_items = os.listdir(extract_dir)
    data_dir  = extract_dir
    if len(sub_items) == 1 and os.path.isdir(os.path.join(extract_dir, sub_items[0])):
        data_dir = os.path.join(extract_dir, sub_items[0])

    # Step 2: Load & extract HOG features
    print("\n[2/5] Loading images and extracting HOG features ...")
    X, y = load_dataset(data_dir)
    print(f"      Total usable samples: {len(X)}")

    if len(X) == 0:
        print("[ERROR] No images loaded. Check zip structure.")
        sys.exit(1)

    # Step 3: Scale features
    print("\n[3/5] Scaling features ...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 4: Train / test split + train SVM
    print("\n[4/5] Training SVM (RBF kernel) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train, y_train)

    # Step 5: Evaluate
    print("\n[5/5] Evaluating model ...")
    y_pred = svm.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    print("\n" + "="*55)
    print("           EVALUATION RESULTS")
    print("="*55)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Cancer", "Cancer"]))
    print("  Confusion Matrix:")
    print(f"                   Predicted")
    print(f"                Non-Cancer  Cancer")
    print(f"  Actual Non-Cancer   {cm[0][0]:>5}    {cm[0][1]:>5}")
    print(f"  Actual Cancer       {cm[1][0]:>5}    {cm[1][1]:>5}")
    print("="*55)

    # Save model & scaler
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(svm, f)
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n  ✅ Model saved  → {MODEL_SAVE_PATH}")
    print(f"  ✅ Scaler saved → {SCALER_SAVE_PATH}")
    print("\n  🎉 Phase 1 complete! Now run Phase 2:")
    print("     python -m streamlit run phase2_app.py")
    print("="*55 + "\n")


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 — Train Oral Cancer SVM")
    parser.add_argument("--zip", required=True, help="Path to dataset zip file")
    args = parser.parse_args()

    if not os.path.exists(args.zip):
        print(f"[ERROR] Zip file not found: {args.zip}")
        sys.exit(1)

    train(args.zip)