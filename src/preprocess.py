import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

def strict_preprocessing(input_path="processed_data/gas_data_full.csv", output_path="processed_data/gas_data_normalized.csv"):
    """
    Normalizes data while STRICTLY preventing Data Leakage.
    Protocol: 
    1. Split Data into Source (Batch 1) and Target (Batches 2-10).
    2. Fit Scaler ONLY on Source.
    3. Apply that Scaler to Target.
    """
    print(f"Loading raw data from {input_path}...")
    
    # Path handling (local vs project root)
    if not os.path.exists(input_path):
        # Fallback if running from root
        input_path = "processed_data/gas_data_full.csv"
        if not os.path.exists(input_path):
             print(f"❌ Error: {input_path} not found. Run data_loader.py first!")
             return

    df = pd.read_csv(input_path)
    
    # 1. Identify Columns
    feat_cols = [c for c in df.columns if 'feat_' in c]
    meta_cols = [c for c in df.columns if 'feat_' not in c]
    
    print(f"Found {len(feat_cols)} features and {len(df)} total samples.")
    
    # 2. SPLIT: Isolate Source vs Target
    # Source = Batch 1
    source_df = df[df['Batch_ID'] == 1].copy()
    # Target = Batches 2 through 10
    target_df = df[df['Batch_ID'] != 1].copy()
    
    print(f"Source Domain (Batch 1): {len(source_df)} samples")
    print(f"Target Domain (Batches 2-10): {len(target_df)} samples")
    
    # 3. FIT: Learn Mean/Std from Source ONLY
    print("\n[CRITICAL STEP] Fitting Scaler on Batch 1 only...")
    scaler = StandardScaler()
    
    # Extract raw numpy array for fitting
    source_features = source_df[feat_cols].values
    scaler.fit(source_features)
    
    print(f"  > Source Mean (Feature 0): {scaler.mean_[0]:.4f}")
    print(f"  > Source Var  (Feature 0): {scaler.var_[0]:.4f}")
    
    # 4. TRANSFORM: Apply to Source AND Target
    print("Applying transform to all data...")
    
    # Transform Source (Will become Mean~0, Std~1)
    source_scaled = scaler.transform(source_features)
    
    # Transform Target (Will NOT be Mean~0. It will retain the drift offset relative to Source)
    target_features = target_df[feat_cols].values
    target_scaled = scaler.transform(target_features)
    
    # 5. VERIFICATION: Check Drift Preservation
    src_mean_after = source_scaled[:, 0].mean()
    tgt_mean_after = target_scaled[:, 0].mean()
    print(f"\n[PHYSICS CHECK]")
    print(f"  > Source Mean After Scaling: {src_mean_after:.4f} (Should be ~0.0)")
    print(f"  > Target Mean After Scaling: {tgt_mean_after:.4f} (Should NOT be 0.0)")
    
    if abs(tgt_mean_after) > 0.1:
        print("  ✅ PASS: Drift Signal is preserved in Target data.")
    else:
        print("  ⚠️ WARNING: Target mean is close to 0. Did you accidentally fit on Target?")

    # 6. REASSEMBLE
    # Put scaled values back into the DataFrames
    source_df[feat_cols] = source_scaled
    target_df[feat_cols] = target_scaled
    
    # Concatenate back into one file (sorted by batch)
    df_final = pd.concat([source_df, target_df]).sort_values(by='Batch_ID')
    
    # 7. SAVE
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    df_final.to_csv(output_path, index=False)
    
    # Save the scaler model (optional, good for reproducibility)
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    joblib.dump(scaler, "checkpoints/source_scaler.pkl")
    
    print(f"\n✅ Success! Saved properly normalized data to {output_path}")

if __name__ == "__main__":
    strict_preprocessing()