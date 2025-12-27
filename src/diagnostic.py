import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import os
import sys

# Import your architecture
sys.path.append(os.path.abspath("."))
from src.models import SiameseEncoder, TaskClassifier, PhysicsHead
from src.data_loader import GasDataset

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"

def load_checkpoint(batch_id):
    """Loads the model state after adapting to a specific batch"""
    fname = f"adapted_model_b{batch_id}.pth"
    if batch_id == 1: fname = "source_model.pth"
    
    path = os.path.join(CHECKPOINT_DIR, fname)
    if not os.path.exists(path):
        print(f"⚠️ Checkpoint {fname} not found.")
        return None, None, None

    # Init structure
    enc = SiameseEncoder(128, 64).to(DEVICE)
    cls = TaskClassifier(64, 6).to(DEVICE)
    phy = PhysicsHead(64).to(DEVICE)
    
    # Load weights
    ckpt = torch.load(path, map_location=DEVICE)
    enc.load_state_dict(ckpt['enc'])
    cls.load_state_dict(ckpt['cls'])
    phy.load_state_dict(ckpt['phy'])
    
    enc.eval(); cls.eval(); phy.eval()
    return enc, cls, phy

def run_diagnostics():
    print("--- STARTING DEEP DIAGNOSTICS ---")
    
    # 1. Load Data
    df = pd.read_csv("processed_data/gas_data_normalized.csv")
    
    metrics = []
    
    # Loop through critical batches (Source, Transition, Crash Points)
    # 1=Source, 2=Good Adapt, 5=Stuck, 8=Crash
    target_batches = [1, 2, 5, 8, 10] 
    
    fig_tsne, axes_tsne = plt.subplots(1, len(target_batches), figsize=(25, 5))
    
    for i, b_id in enumerate(target_batches):
        print(f"Analyzing Model State after Batch {b_id}...")
        
        # Load Model
        enc, cls, phy = load_checkpoint(b_id)
        if enc is None: continue
        
        # Load Data for this batch
        ds = GasDataset(df, batch_id=b_id)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        
        all_z = []
        all_preds = []
        all_labels = []
        all_confs = []
        all_baselines = []
        
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(DEVICE)
                z = enc(x)
                
                # Classifier
                logits = cls(z)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                
                # Physics
                _, base = phy(z)
                
                all_z.append(z.cpu().numpy())
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.numpy())
                all_confs.append(conf.cpu().numpy())
                all_baselines.append(base.cpu().numpy())
                
        # Aggregate
        all_z = np.concatenate(all_z)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_confs = np.concatenate(all_confs)
        all_baselines = np.concatenate(all_baselines)
        
        # --- METRIC 1: PSEUDO-LABEL HEALTH ---
        # How many samples would pass the 0.6 threshold?
        pass_rate = (all_confs > 0.6).mean()
        print(f"  > Batch {b_id} Confidence > 0.6: {pass_rate:.1%} of samples")
        
        # --- METRIC 2: BASELINE CONSISTENCY ---
        avg_base = np.mean(all_baselines)
        print(f"  > Batch {b_id} Avg Baseline Est: {avg_base:.4f}")
        
        # --- PLOT: T-SNE ---
        # We limit samples to speed up T-SNE
        if len(all_z) > 1000:
            idx = np.random.choice(len(all_z), 1000, replace=False)
            z_sub = all_z[idx]
            y_sub = all_labels[idx]
        else:
            z_sub = all_z
            y_sub = all_labels
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        z_2d = tsne.fit_transform(z_sub)
        
        ax = axes_tsne[i] if len(target_batches) > 1 else axes_tsne
        sns.scatterplot(x=z_2d[:,0], y=z_2d[:,1], hue=y_sub, palette="tab10", s=20, ax=ax, legend=False)
        ax.set_title(f"Batch {b_id} Latent Space\nConf > 0.6: {pass_rate:.1%}")
        ax.grid(True, alpha=0.3)
        
        metrics.append({
            'Batch': b_id,
            'Pass_Rate': pass_rate,
            'Avg_Baseline': avg_base
        })

    plt.tight_layout()
    plt.savefig("diagnostics_tsne.png")
    print("\nSaved diagnostics_tsne.png")
    
    # Save Metrics CSV
    pd.DataFrame(metrics).to_csv("diagnostics_metrics.csv", index=False)
    print("Saved diagnostics_metrics.csv")

if __name__ == "__main__":
    run_diagnostics()