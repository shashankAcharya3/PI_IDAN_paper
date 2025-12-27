import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def run_eda(csv_path="processed_data/gas_data_full.csv"):
    # 1. Setup
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run data_loader.py first.")
        return

    if not os.path.exists("plots"):
        os.makedirs("plots")

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 2. FEATURE ANALYSIS: Check the "Scale" problem
    # We look at Feature 0 (Steady State) vs Feature 2 (Transient EMA)
    print("\n--- STATS CHECK ---")
    feat_cols = [c for c in df.columns if 'feat_' in c]
    print(f"Feature 0 (Resistance) Mean: {df['feat_0'].mean():.2f}")
    print(f"Feature 2 (EMA alpha=0.001) Mean: {df['feat_2'].mean():.2f}")
    
    if df['feat_0'].mean() > 1000 and df['feat_2'].mean() < 10:
        print(">> VERDICT: Huge Scale Difference Detected. Normalization is MANDATORY.")

    # 3. VISUALIZATION A: The "Drift" (PCA)
    # We take Batch 1 (Source) and Batch 10 (Target)
    print("\nGenerating Drift Plot (PCA)...")
    batch1 = df[df['Batch_ID'] == 1]
    batch10 = df[df['Batch_ID'] == 10]
    
    combined = pd.concat([batch1, batch10])
    features = combined[feat_cols].values
    labels = combined['Batch_ID'].values
    gas_types = combined['Gas_Class'].values

    # Compress 128D -> 2D
    pca = PCA(n_components=2)
    embedded = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=embedded[:,0], y=embedded[:,1], hue=labels, palette={1:'blue', 10:'red'}, alpha=0.6)
    plt.title("Visual Proof of Sensor Drift (Batch 1 vs Batch 10)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/1_drift_proof_pca.png")
    plt.close()
    print("Saved plots/1_drift_proof_pca.png")

    # 4. VISUALIZATION B: Class Imbalance
    print("Generating Class Distribution Plot...")
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Batch_ID', hue='Gas_Class')
    plt.title("Class Distribution per Batch (Check for Missing Gases)")
    plt.ylabel("Number of Samples")
    plt.legend(title='Gas Class', loc='upper right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig("plots/2_class_distribution.png")
    plt.close()
    print("Saved plots/2_class_distribution.png")

if __name__ == "__main__":
    run_eda()