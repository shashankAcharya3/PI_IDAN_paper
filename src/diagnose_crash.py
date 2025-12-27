import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
import sys

# Import architecture
from models import SiameseEncoder, TaskClassifier

def diagnose_batch2_crash():
    print("--- DIAGNOSING BATCH 2 CRASH ---")
    
    # 1. Load Data
    csv_path = "processed_data/gas_data_normalized.csv"
    if not os.path.exists(csv_path): csv_path = "gas_data_normalized.csv"
    df = pd.read_csv(csv_path)
    
    # Get Batch 2 Data
    b2_df = df[df['Batch_ID'] == 2]
    feat_cols = [c for c in df.columns if 'feat_' in c]
    X_b2 = b2_df[feat_cols].values.astype(np.float32)
    y_b2 = b2_df['Gas_Class'].values - 1 # Map 1-6 to 0-5
    
    print(f"Loaded Batch 2: {len(X_b2)} samples")

    # 2. Load Models
    device = torch.device("cpu") # CPU is fine for inference
    
    # Model A: Source Only (The Good One)
    enc_source = SiameseEncoder().to(device)
    cls_source = TaskClassifier().to(device)
    
    try:
        ckpt_source = torch.load("checkpoints/source_model.pth", map_location=device)
        enc_source.load_state_dict(ckpt_source['enc'])
        cls_source.load_state_dict(ckpt_source['cls'])
        print("Loaded Source Model (Before Crash)")
    except Exception as e:
        print(f"Error loading source model: {e}")
        return

    # Model B: Adapted (The Crashed One)
    enc_adapt = SiameseEncoder().to(device)
    
    try:
        ckpt_adapt = torch.load("checkpoints/adapted_model_b2.pth", map_location=device)
        enc_adapt.load_state_dict(ckpt_adapt['enc'])
        print("Loaded Adapted Model (After Crash)")
    except:
        print("⚠️ Could not load adapted_model_b2.pth. Did it save?")
        enc_adapt = None

    # 3. Run Inference & Collect Latents
    enc_source.eval(); cls_source.eval()
    if enc_adapt: enc_adapt.eval()
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_b2).to(device) # Add channel dim for TCN
        
        # A. Before Crash
        # Use _get_z logic manually: TCN returns z (if modified) or tuple
        out_src = enc_source(X_tensor)
        z_src = out_src[0] if isinstance(out_src, tuple) else out_src
        
        logits_src = cls_source(z_src)
        probs_src = torch.softmax(logits_src, dim=1)
        conf_src, _ = torch.max(probs_src, dim=1)
        
        # B. After Crash
        if enc_adapt:
            out_adapt = enc_adapt(X_tensor)
            z_adapt = out_adapt[0] if isinstance(out_adapt, tuple) else out_adapt
            
            logits_adapt = cls_source(z_adapt) # Classifier is shared/frozen
            probs_adapt = torch.softmax(logits_adapt, dim=1)
            conf_adapt, _ = torch.max(probs_adapt, dim=1)

    # 4. PLOTTING
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # T-SNE 1: Before Crash
    print("Computing t-SNE (Before)...")
    tsne = TSNE(n_components=2, random_state=42)
    z_src_2d = tsne.fit_transform(z_src.numpy())
    
    sns.scatterplot(x=z_src_2d[:,0], y=z_src_2d[:,1], hue=y_b2, palette='tab10', ax=axes[0,0], legend='full')
    axes[0,0].set_title("Feature Space BEFORE Adaptation (Batch 2)\n(Should see clear clusters)")
    
    # Histogram 1: Confidence Before
    sns.histplot(conf_src.numpy(), bins=20, ax=axes[0,1], color='green')
    axes[0,1].set_title("Confidence Distribution BEFORE Adaptation\n(Should be high)")
    axes[0,1].axvline(0.8, color='red', linestyle='--', label='Threshold')

    if enc_adapt:
        # T-SNE 2: After Crash
        print("Computing t-SNE (After)...")
        z_adapt_2d = tsne.fit_transform(z_adapt.numpy())
        
        sns.scatterplot(x=z_adapt_2d[:,0], y=z_adapt_2d[:,1], hue=y_b2, palette='tab10', ax=axes[1,0], legend=False)
        axes[1,0].set_title("Feature Space AFTER Adaptation (Batch 2)\n(If blobs merged = Mode Collapse)")
        
        # Histogram 2: Confidence After
        sns.histplot(conf_adapt.numpy(), bins=20, ax=axes[1,1], color='orange')
        axes[1,1].set_title("Confidence Distribution AFTER Adaptation\n(Shift left = Confusion)")
        axes[1,1].axvline(0.8, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig("crash_diagnosis.png")
    plt.show()
    print("✅ Diagnosis Plot saved to crash_diagnosis.png")

if __name__ == "__main__":
    diagnose_batch2_crash()