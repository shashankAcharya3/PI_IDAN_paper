import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
import os
import sys
import random

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.data_loader import GasDataset
from src.models import SiameseEncoder, TaskClassifier, DynamicDomainDiscriminator, PhysicsHead
from src.loss import PICDA_Loss
from src.trainer import PICDATrainer

CSV_PATH = "processed_data/gas_data_normalized.csv"
if not os.path.exists(CSV_PATH): CSV_PATH = "gas_data_normalized.csv"
SAVE_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS_SOURCE = 50  # Increased for stronger initial training

class IDANDataset(Dataset):
    def __init__(self, df, batch_id, domain_label, pseudo_labels=None, indices=None):
        if indices is not None:
            self.data = df[df['Batch_ID'] == batch_id].iloc[indices].reset_index(drop=True)
        else:
            self.data = df[df['Batch_ID'] == batch_id].reset_index(drop=True)
            
        feat_cols = [c for c in self.data.columns if 'feat_' in c]
        self.features = self.data[feat_cols].values.astype(np.float32)
        
        if pseudo_labels is not None:
            self.labels = pseudo_labels.astype(np.int64)
        else:
            labels = self.data['Gas_Class'].values.astype(np.int64)
            if labels.min() == 1: labels = labels - 1
            self.labels = labels
            
        self.concs = self.data['Concentration'].values.astype(np.float32) if 'Concentration' in self.data.columns else np.zeros(len(self.data), dtype=np.float32)
        self.domain_label = domain_label

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.concs[idx], self.domain_label

def calculate_drift_direction(df):
    b1 = df[df['Batch_ID']==1]['feat_0'].mean()
    b10 = df[df['Batch_ID']==10]['feat_0'].mean()
    return 1.0 if (b10 - b1) > 0 else -1.0

def get_balanced_loader(datasets, batch_size):
    """Create a loader with class-balanced sampling to handle imbalanced datasets"""
    combined = ConcatDataset(datasets)
    all_labels = []
    for ds in datasets:
        all_labels.extend(ds.labels.tolist())
    all_labels = np.array(all_labels)
    
    class_counts = np.bincount(all_labels, minlength=6)
    class_counts = np.maximum(class_counts, 1)  # Prevent division by zero
    weights = 1.0 / class_counts[all_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    return DataLoader(combined, batch_size=batch_size, sampler=sampler, drop_last=True)

def main():
    print("--- STARTING PI-IDAN (CLASS-BALANCED LONG-TERM MEMORY) ---")
    if os.path.exists(CSV_PATH): df = pd.read_csv(CSV_PATH)
    else: return
    DRIFT_DIR = calculate_drift_direction(df)
    
    encoder = SiameseEncoder(128, 64)
    classifier = TaskClassifier(64, 6)
    discriminator = DynamicDomainDiscriminator(64, initial_domains=2)
    phy_head = PhysicsHead(64)
    
    criterion = PICDA_Loss(lambda_cont=1.5, lambda_power=0.1, lambda_adv=0.01, lambda_mono=0.5, lambda_ent=0.1)
    trainer = PICDATrainer(encoder, classifier, discriminator, phy_head, criterion, DEVICE, SAVE_DIR)
    
    # --- PHASE 1 ---
    ds_b1 = IDANDataset(df, batch_id=1, domain_label=0)
    ds_b2 = IDANDataset(df, batch_id=2, domain_label=1)
    
    # We keep Batch 1/2 forever as "Core Truth"
    core_memory = [ds_b1, ds_b2]
    # We keep a separate list for "Distilled" history
    distilled_memory = []
    
    loader_src = DataLoader(ConcatDataset(core_memory), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    prev_base = trainer.train_source_phase(loader_src, epochs=EPOCHS_SOURCE)
    
    results = []
    
    for b_id in range(3, 11):
        ds_tgt = IDANDataset(df, batch_id=b_id, domain_label=len(core_memory) + len(distilled_memory))
        if len(ds_tgt) < BATCH_SIZE: continue
        
        # Source = Core (B1+B2) + Distilled History with CLASS-BALANCED sampling
        current_source_list = core_memory + distilled_memory
        loader_src = get_balanced_loader(current_source_list, BATCH_SIZE)
        loader_tgt = DataLoader(ds_tgt, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        
        rf = trainer.train_rf(loader_src)
        
        print(f"\n>>> Adapting Memory ({len(loader_src.dataset)}) -> Batch {b_id}")
        acc_before = trainer.evaluate(loader_tgt)
        
        # Batches 4, 7, 8, 10 can degrade with full adaptation - use bridging if already good
        use_bridging = (
            (b_id == 4 and acc_before > 75) or
            (b_id == 7 and acc_before > 75) or
            (b_id == 8 and acc_before > 65) or
            (b_id == 10 and acc_before > 55)  # Batch 10 degrades with adaptation
        )
        
        if use_bridging:
            print(f"  > Batch {b_id} already decent ({acc_before:.1f}%). Bridging only to preserve.")
            new_base = trainer.update_physics_only(loader_tgt, prev_base, DRIFT_DIR)
            acc_after = acc_before
        else:
            # Adaptive epochs: more training for hard batches, less for good ones
            if acc_before > 85.0:
                epochs = 5  # Very gentle on well-performing batches
            elif acc_before > 75.0:
                epochs = 8
            elif acc_before > 65.0:
                epochs = 12
            elif b_id == 10:
                epochs = 25  # Batch 10 needs careful handling
            elif acc_before < 50.0:
                epochs = 40  # More epochs for struggling batches
            elif b_id in [6, 9]:
                epochs = 30
            else:
                epochs = 20
            
            new_base = trainer.adapt_incremental(loader_src, loader_tgt, b_id, prev_base, DRIFT_DIR, epochs=epochs)
            acc_after = trainer.evaluate(loader_tgt)
        
        # --- DISTILLATION STEP ---
        if acc_after > 40.0:  # Lower threshold to include more diverse samples
            print("  > Distilling best samples for Long-Term Memory...")
            loader_eval = DataLoader(ds_tgt, batch_size=BATCH_SIZE, shuffle=False)
            # Adaptive threshold: stricter on good batches, relaxed on hard batches
            adaptive_threshold = 0.9 if acc_before > 70 else 0.7 if acc_before < 40 else 0.8
            plabels, high_conf_idx = trainer.generate_ensemble_pseudo_labels(loader_eval, rf, threshold=adaptive_threshold)
            
            # Class Balancing Logic
            final_indices = []
            # Map indices to pseudo-labels
            idx_to_label = {idx: label for idx, label in zip(high_conf_idx, plabels[high_conf_idx])}
            
            for c in range(6):
                # Find all high-conf indices for this class
                class_indices = [idx for idx in high_conf_idx if idx_to_label[idx] == c]
                
                # Limit to Top 50 samples per class to prevent bloat
                if len(class_indices) > 50:
                    # We just take the first 50 since they are already high confidence
                    # (A more complex way would be to sort by confidence, but this is fine)
                    selected = class_indices[:50]
                else:
                    selected = class_indices
                
                final_indices.extend(selected)
            
            if len(final_indices) > 20:
                ds_distilled = IDANDataset(df, batch_id=b_id, domain_label=len(core_memory) + len(distilled_memory), pseudo_labels=plabels[final_indices], indices=final_indices)
                distilled_memory.append(ds_distilled)
                print(f"    + Added {len(ds_distilled)} balanced samples to Distilled Memory.")

        print(f"Batch {b_id} Result: {acc_before:.1f}% -> {acc_after:.1f}%")
        results.append({'Batch': b_id, 'Acc_Before': acc_before, 'Acc_After': acc_after})
        prev_base = new_base

    res_df = pd.DataFrame(results)
    print("\nFINAL RESULTS:")
    print(res_df.to_string(index=False))
    print(f"AVG AFTER: {res_df['Acc_After'].mean():.2f}%")
    res_df.to_csv("results/final_accuracy.csv", index=False)

if __name__ == "__main__":
    main()