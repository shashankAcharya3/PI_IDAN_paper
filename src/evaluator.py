"""
Evaluation Metrics Module for PI-CDA
Computes Precision, Recall, F1 and confusion matrix for sensor drift classification
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score


def compute_metrics(y_true, y_pred, num_classes=6):
    """
    Compute detailed evaluation metrics.
    
    Args:
        y_true: Ground truth labels (list or array)
        y_pred: Predicted labels (list or array)
        num_classes: Number of classes (default: 6 gas types)
    
    Returns:
        dict with accuracy, precision, recall, f1 (macro), and confusion matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # Per-class and macro-averaged metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )
    
    # Macro averages
    precision_macro = np.mean(precision) * 100
    recall_macro = np.mean(recall) * 100
    f1_macro = np.mean(f1) * 100
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    return {
        'accuracy': accuracy,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'precision_per_class': precision * 100,
        'recall_per_class': recall * 100,
        'f1_per_class': f1 * 100,
        'support_per_class': support,
        'confusion_matrix': cm,
        'f1_macro': f1_macro / 100  # For unit test compatibility
    }


def evaluate_detailed(encoder, classifier, loader, device, num_classes=6):
    """
    Run model evaluation and compute complete metrics.
    
    Args:
        encoder: Feature encoder model
        classifier: Classification head
        loader: DataLoader for evaluation data
        device: torch device
        num_classes: Number of classes
    
    Returns:
        dict with all evaluation metrics
    """
    encoder.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            z = encoder(x)
            logits = classifier(z)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    return compute_metrics(all_labels, all_preds, num_classes)


def format_metrics_table(metrics_dict):
    """Format metrics as a printable string table."""
    lines = [
        f"Accuracy:  {metrics_dict['accuracy']:.2f}%",
        f"Precision: {metrics_dict['precision']:.2f}%",
        f"Recall:    {metrics_dict['recall']:.2f}%",
        f"F1-Score:  {metrics_dict['f1']:.2f}%",
        "",
        "Per-Class Breakdown:",
    ]
    
    gas_names = ['Gas 1', 'Gas 2', 'Gas 3', 'Gas 4', 'Gas 5', 'Gas 6']
    lines.append(f"{'Class':<8} {'PR%':>8} {'RC%':>8} {'F1%':>8} {'Support':>8}")
    lines.append("-" * 42)
    
    for i, name in enumerate(gas_names):
        pr = metrics_dict['precision_per_class'][i]
        rc = metrics_dict['recall_per_class'][i]
        f1 = metrics_dict['f1_per_class'][i]
        sup = metrics_dict['support_per_class'][i]
        lines.append(f"{name:<8} {pr:>8.1f} {rc:>8.1f} {f1:>8.1f} {sup:>8}")
    
    return "\n".join(lines)
