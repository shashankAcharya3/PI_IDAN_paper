import torch
import torch.nn as nn
import torch.nn.functional as F

class PICDA_Loss(nn.Module):
    def __init__(self, lambda_cont=1.0, lambda_power=0.1, lambda_adv=0.1, lambda_mono=1.0, lambda_ent=0.1):
        super().__init__()
        self.lambda_cont = lambda_cont
        self.lambda_power = lambda_power
        self.lambda_adv = lambda_adv
        self.lambda_mono = lambda_mono
        self.lambda_ent = lambda_ent 
        
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.class_weights = None  # For class-weighted loss

    def set_class_weights(self, labels):
        """Compute inverse frequency weights from label distribution"""
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu()
        unique, counts = torch.unique(torch.as_tensor(labels), return_counts=True)
        weights = torch.zeros(6)  # 6 classes
        for u, c in zip(unique, counts):
            weights[u] = 1.0 / c.float()
        weights = weights / weights.sum() * 6  # Normalize
        self.class_weights = weights

    def task_loss(self, preds, targets):
        if self.class_weights is not None:
            weighted_ce = nn.CrossEntropyLoss(weight=self.class_weights.to(preds.device))
            return weighted_ce(preds, targets)
        return self.ce(preds, targets)

    def entropy_loss(self, logits):
        p = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(p * log_p, dim=1)
        return entropy.mean()

    def contrastive_loss(self, features, labels, temperature=0.07):
        device = features.device
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / temperature
        
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(features.size(0)).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
        
        # --- CRITICAL FIX: Normalize by positive pairs count ---
        pos_pairs_count = mask.sum(1) + 1e-6
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_pairs_count
        
        return -mean_log_prob_pos.mean()

    def power_law_loss(self, sensitivity_mag, concentrations, w, b):
        log_conc = torch.log(concentrations.view(-1, 1) + 1e-6)
        target_mag = w * log_conc + b
        return self.mse(sensitivity_mag, target_mag)

    def monotonicity_loss(self, current_base, prev_base, drift_dir):
        delta = current_base.mean() - prev_base
        return torch.relu(-delta * drift_dir)

    def adversarial_loss(self, preds, target_is_real):
        targets = torch.ones_like(preds) if target_is_real else torch.zeros_like(preds)
        return self.bce(preds, targets)
    
    # --- ADVANCED PHYSICS-INFORMED LOSSES ---
    
    def prototype_contrastive_loss(self, features, labels, prototypes, temperature=0.1):
        """
        Pull features toward their class prototype (centroid).
        Prototypes are maintained as a memory bank.
        """
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        
        # Compute similarity to all prototypes
        sim = torch.matmul(features, prototypes.T) / temperature  # [N, num_classes]
        
        # Cross entropy against true labels
        return F.cross_entropy(sim, labels)
    
    def physics_consistency_loss(self, z_s, z_t, y_s, plabels, mask, physics_head):
        """
        Enforce physics consistency: samples of the same class should have
        similar physics properties (baseline, sensitivity) across domains.
        """
        loss = torch.tensor(0.0, device=z_s.device)
        count = 0
        
        for c in range(6):
            src_idx = (y_s == c)
            tgt_idx = mask & (plabels == c)
            
            if src_idx.sum() > 0 and tgt_idx.sum() > 0:
                _, base_s = physics_head(z_s[src_idx])
                _, base_t = physics_head(z_t[tgt_idx])
                
                # Same class should have similar baseline across domains
                loss = loss + F.mse_loss(base_s.mean(), base_t.mean())
                count += 1
        
        return loss / max(count, 1)
    
    def temporal_ensemble_loss(self, current_preds, ema_preds, temperature=0.5):
        """
        Consistency regularization between current model and EMA (temporal ensemble).
        Encourages stable predictions over training.
        """
        current_probs = F.softmax(current_preds / temperature, dim=1)
        ema_probs = F.softmax(ema_preds / temperature, dim=1)
        return F.mse_loss(current_probs, ema_probs.detach())