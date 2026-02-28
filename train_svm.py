import os, numpy as np
from PIL import Image
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

IMG_SIZE = (64, 64)
CANCER_DIR     = "/home/claude/dataset/Multispectral_MSI_Dataset/MSI_Cancer"
NON_CANCER_DIR = "/home/claude/dataset/Multispectral_MSI_Dataset/MSI_Non Cancer"
N_PCA = 100
RS = 42

def load_images(folder, label):
    X, y = [], []
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpeg','.jpg','.png'))]
    print(f"  Loading {len(files)} images from: {os.path.basename(folder)}")
    for fname in files:
        try:
            img = Image.open(os.path.join(folder, fname)).convert('RGB').resize(IMG_SIZE)
            X.append(np.array(img, dtype=np.float32).flatten() / 255.0)
            y.append(label)
        except: pass
    return X, y

print("="*60)
print("  PHASE 1 - SVM Cancer Classifier (MSI Thermal Images)")
print("="*60)

print("\n[1/5] Loading dataset...")
Xc, yc = load_images(CANCER_DIR, 1)
Xn, yn = load_images(NON_CANCER_DIR, 0)
X = np.array(Xc + Xn)
y = np.array(yc + yn)
print(f"  Total: {len(X)} | Cancer: {sum(y==1)} | Non-Cancer: {sum(y==0)}")

print("\n[2/5] Preprocessing - Scaling + PCA...")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)
pca = PCA(n_components=N_PCA, random_state=RS)
X_tr_p = pca.fit_transform(X_tr_sc)
X_te_p = pca.transform(X_te_sc)
expl = pca.explained_variance_ratio_.sum()*100
print(f"  PCA: {N_PCA} components -> {expl:.1f}% variance explained")

print("\n[3/5] Training SVM (RBF kernel, C=10)...")
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced', random_state=RS)
svm.fit(X_tr_p, y_tr)
print("  Training complete!")

print("\n[4/5] Evaluating...")
y_pred = svm.predict(X_te_p)
y_prob = svm.predict_proba(X_te_p)[:,1]
acc = accuracy_score(y_te, y_pred)
auc = roc_auc_score(y_te, y_prob)
print(f"\n  Test Accuracy : {acc*100:.2f}%")
print(f"  ROC-AUC Score : {auc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_te, y_pred, target_names=['Non-Cancer','Cancer']))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
cv_sc = cross_val_score(svm, X_tr_p, y_tr, cv=cv, scoring='accuracy')
print(f"  5-Fold CV: {cv_sc.mean()*100:.2f}% +/- {cv_sc.std()*100:.2f}%")

print("\n[5/5] Saving plots and models...")
fig, axes = plt.subplots(1,3,figsize=(16,5))
fig.suptitle("Phase 1 - SVM Cancer Classifier (MSI Thermal Images)", fontsize=13, fontweight='bold')
cm = confusion_matrix(y_te, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Non-Cancer','Cancer']).plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Confusion Matrix")
axes[1].bar(['Non-Cancer','Cancer'],[sum(y==0),sum(y==1)], color=['steelblue','tomato'])
axes[1].set_title("Class Distribution"); axes[1].set_ylabel("Count")
for i,v in enumerate([sum(y==0),sum(y==1)]): axes[1].text(i,v+2,str(v),ha='center',fontweight='bold')
axes[2].plot(np.cumsum(pca.explained_variance_ratio_)*100, color='purple')
axes[2].axhline(y=expl, color='red', linestyle='--', alpha=0.6, label=f'{expl:.1f}% @ {N_PCA} comps')
axes[2].set_title("PCA Cumulative Variance"); axes[2].set_xlabel("Components"); axes[2].set_ylabel("Variance (%)"); axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/phase1_results.png", dpi=150, bbox_inches='tight')
plt.close()

joblib.dump(svm,    "/home/claude/svm_cancer_model.pkl")
joblib.dump(scaler, "/home/claude/scaler.pkl")
joblib.dump(pca,    "/home/claude/pca.pkl")
print("  Models saved: svm_cancer_model.pkl, scaler.pkl, pca.pkl")
print("\n" + "="*60)
print("  Phase 1 Complete!")
print("="*60)
