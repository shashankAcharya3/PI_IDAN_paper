import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score
import scipy.stats as stats
import os
import sys

# Import architecture
sys.path.append(os.path.abspath("."))
from src.models import SiameseEncoder, TaskClassifier, PhysicsHead
from src.data_loader import GasDataset

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
PLOT_DIR = "diagnostic_plots"

if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)

def load_adapted_model(batch_id):
    """Loads the specific model adapted for a given batch"""
    fname = f"adapted_model_b{batch_id}.pth"
    path = os.path.join(CHECKPOINT_DIR, fname)
    
    if not os.path.exists(path):
        print(f"⚠️ Warning: {fname} not found. Skipping.")
        return None, None, None

    enc = SiameseEncoder(128, 64).to(DEVICE)
    cls = TaskClassifier(64, 6).to(DEVICE)
    phy = PhysicsHead(64).to(DEVICE)
    
    ckpt = torch.load(path, map_location=DEVICE)
    enc.load_state_dict(ckpt['enc'])
    cls.load_state_dict(ckpt['cls'])
    phy.load_state_dict(ckpt['phy'])
    
    enc.eval(); cls.eval(); phy.eval()
    return enc, cls, phy

def analyze_model_health():
    print("--- STARTING COMPREHENSIVE DIAGNOSIS ---")
    df = pd.read_csv("processed_data/gas_data_normalized.csv")
    
    report_data = []
    
    # We analyze specific interesting batches
    # 2 (Early), 6 (The Shift), 8 (The Toxic One), 10 (The End)
    target_batches = [2, 4, 6, 8, 9, 10]
    
    for b_id in target_batches:
        print(f"\n🔍 Diagnosing Batch {b_id}...")
        enc, cls, phy = load_adapted_model(b_id)
        if enc is None: continue
        
        ds = GasDataset(df, batch_id=b_id)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        
        all_z = []
        all_preds = []
        all_labels = []
        all_confs = []
        all_base = []
        all_mags = []
        all_concs = []
        
        with torch.no_grad():
            for x, y, c in loader:
                x = x.to(DEVICE)
                z = enc(x)
                
                # Classifier
                logits = cls(z)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                
                # Physics
                mag, base = phy(z)
                
                all_z.append(z.cpu().numpy())
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.numpy())
                all_confs.append(conf.cpu().numpy())
                all_base.append(base.cpu().numpy())
                all_mags.append(mag.cpu().numpy())
                all_concs.append(c.numpy())

        # Concatenate
        all_z = np.concatenate(all_z)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_confs = np.concatenate(all_confs)
        all_base = np.concatenate(all_base).flatten()
        all_mags = np.concatenate(all_mags).flatten()
        all_concs = np.concatenate(all_concs)

        # --- ANALYSIS 1: ACCURACY & CONFIDENCE ---
        acc = accuracy_score(all_labels, all_preds) * 100
        avg_conf = np.mean(all_confs)
        # Check "Confident Wrong" rate (Hallucination)
        wrong_mask = all_preds != all_labels
        conf_wrong = np.mean(all_confs[wrong_mask]) if wrong_mask.sum() > 0 else 0
        
        # --- ANALYSIS 2: PHYSICS HEALTH ---
        # Correlation between Magnitude |z| and Log(Concentration)
        # Should be high positive correlation
        log_conc = np.log(all_concs + 1e-6)
        r_physics, _ = stats.pearsonr(all_mags, log_conc)
        
        avg_baseline = np.mean(all_base)
        
        print(f"  > Accuracy: {acc:.2f}%")
        print(f"  > Avg Confidence: {avg_conf:.3f} | Conf on Errors: {conf_wrong:.3f}")
        print(f"  > Physics Correlation (Power Law): {r_physics:.3f} (Target > 0.8)")
        print(f"  > Estimated Baseline: {avg_baseline:.4f}")

        # --- PLOT 1: CONFUSION MATRIX ---
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(all_labels, all_preds, normalize='true')
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f"Batch {b_id} Confusion Matrix (Acc {acc:.1f}%)")
        plt.ylabel("True Class"); plt.xlabel("Predicted Class")
        plt.savefig(f"{PLOT_DIR}/b{b_id}_confusion.png")
        plt.close()
        
        # --- PLOT 2: T-SNE (Feature Space) ---
        # Subsample for speed
        if len(all_z) > 1000:
            idx = np.random.choice(len(all_z), 1000, replace=False)
            z_sub = all_z[idx]; y_sub = all_labels[idx]
        else:
            z_sub = all_z; y_sub = all_labels
            
        tsne = TSNE(n_components=2, random_state=42)
        z_2d = tsne.fit_transform(z_sub)
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=z_2d[:,0], y=z_2d[:,1], hue=y_sub, palette="tab10", legend='full')
        plt.title(f"Batch {b_id} Feature Space (t-SNE)")
        plt.savefig(f"{PLOT_DIR}/b{b_id}_tsne.png")
        plt.close()
        
        report_data.append({
            "Batch": b_id, "Accuracy": acc, 
            "Physics_R": r_physics, "Baseline": avg_baseline,
            "Conf_Correct": np.mean(all_confs[~wrong_mask]),
            "Conf_Wrong": conf_wrong
        })

    # Save summary
    pd.DataFrame(report_data).to_csv("diagnostic_report.csv", index=False)
    print(f"\n✅ Diagnosis Complete. Check '{PLOT_DIR}' for images and 'diagnostic_report.csv'.")

if __name__ == "__main__":
    analyze_model_health()