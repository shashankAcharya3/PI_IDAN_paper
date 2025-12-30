import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import copy
from sklearn.ensemble import RandomForestClassifier

class PICDATrainer:
    def __init__(self, encoder, classifier, discriminator, physics_head, criterion, device, save_dir, latent_dim=64, num_classes=6):
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.discriminator = discriminator.to(device)
        self.physics_head = physics_head.to(device)
        self.criterion = criterion.to(device)
        self.device = device
        self.save_dir = save_dir
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        
        # Mean Teacher: EMA copies of encoder and classifier
        self.ema_encoder = copy.deepcopy(encoder).to(device)
        self.ema_classifier = copy.deepcopy(classifier).to(device)
        for p in self.ema_encoder.parameters(): p.requires_grad = False
        for p in self.ema_classifier.parameters(): p.requires_grad = False
        
        # Prototype Memory Bank: class centroids for prototype-based contrastive learning
        self.prototypes = torch.zeros(num_classes, latent_dim).to(device)
        self.prototype_counts = torch.zeros(num_classes).to(device)
    
    def _update_prototypes(self, z, labels, momentum=0.9):
        """Update class prototypes with exponential moving average"""
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    class_mean = z[mask].mean(0)
                    if self.prototype_counts[c] == 0:
                        self.prototypes[c] = class_mean
                    else:
                        self.prototypes[c] = momentum * self.prototypes[c] + (1 - momentum) * class_mean
                    self.prototype_counts[c] += 1

    def _update_ema(self, alpha=0.999):
        """Exponential Moving Average update for Mean Teacher"""
        for ema_p, p in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            ema_p.data.mul_(alpha).add_(p.data, alpha=1 - alpha)
        for ema_p, p in zip(self.ema_classifier.parameters(), self.classifier.parameters()):
            ema_p.data.mul_(alpha).add_(p.data, alpha=1 - alpha)

    def _mixup(self, x, y, alpha=0.2):
        """Mixup data augmentation for robustness"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _freeze_bn_stats(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm1d): m.eval()

    def _get_z(self, x):
        out = self.encoder(x)
        if isinstance(out, tuple): return out[0]
        return out

    # --- PHASE 1: ENHANCED SOURCE TRAINING (Batch 1 + 2) ---
    def train_source_phase(self, source_loader, epochs=20, lr=0.001):
        print(f"\n[PHASE 1] Training Initial Source Model (Batch 1 + 2)...")
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()) + list(self.physics_head.parameters()) + list(self.discriminator.parameters()), 
            lr=lr
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        self.encoder.train(); self.classifier.train(); self.physics_head.train(); self.discriminator.train()
        
        final_base = 0.0
        for epoch in range(epochs):
            accum_base = []
            for batch_data in source_loader:
                x, y, c = batch_data[0], batch_data[1], batch_data[2]
                x, y, c = x.to(self.device), y.to(self.device), c.to(self.device)
                
                # Mixup augmentation for robustness
                mixed_x, y_a, y_b, lam = self._mixup(x, y, alpha=0.2)
                
                optimizer.zero_grad()
                z = self.encoder(mixed_x)
                pred = self.classifier(z)
                sens, base = self.physics_head(z)
                
                # Mixup loss
                task_loss = lam * self.criterion.task_loss(pred, y_a) + (1 - lam) * self.criterion.task_loss(pred, y_b)
                
                # Also train on clean data for contrastive
                z_clean = self.encoder(x)
                
                loss = task_loss + \
                       self.criterion.lambda_cont * self.criterion.contrastive_loss(z_clean, y) + \
                       self.criterion.lambda_power * self.criterion.power_law_loss(sens, c, self.physics_head.w, self.physics_head.b)
                
                loss.backward()
                optimizer.step()
                self._update_ema()  # Update Mean Teacher
                accum_base.append(base.mean().item())
            
            scheduler.step()
            final_base = np.mean(accum_base)
            if (epoch+1)%5==0: print(f"  Epoch {epoch+1}: Base {final_base:.4f}")
            
        self.save_checkpoint("source_model.pth")
        return final_base

    # --- PHASE 2: INCREMENTAL ADAPTATION ---
    def adapt_incremental(self, source_loader, target_loader, batch_id, prev_baseline, drift_dir, epochs=10, lr=0.0001):
        print(f"\n[PHASE 2] Incremental Adaptation to Batch {batch_id}...")
        
        # 1. Expand Discriminator for the new batch
        self.discriminator.add_new_domain()
        self.discriminator.to(self.device)
        
        # 2. Enable slow classifier fine-tuning (prevent forgetting while allowing adaptation)
        for p in self.classifier.parameters(): p.requires_grad = True
        self.physics_head.w.requires_grad = False; self.physics_head.b.requires_grad = False
        
        # 3. Optimizer with different learning rates (classifier 10x slower)
        optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.discriminator.parameters(), 'lr': lr},
            {'params': self.physics_head.net.parameters(), 'lr': lr},
            {'params': self.classifier.parameters(), 'lr': lr * 0.1}  # 10x slower to prevent forgetting
        ])
        
        self.encoder.train(); self._freeze_bn_stats()
        self.discriminator.train(); self.physics_head.train(); self.classifier.train()
        
        # Domain Labels:
        # Source Data comes from domains {0, 1, ..., K-1}
        # Target Data (Current Batch) is domain {K}
        target_domain_idx = self.discriminator.num_domains - 1
        
        for epoch in range(epochs):
            min_len = min(len(source_loader), len(target_loader))
            iter_src = iter(source_loader); iter_tgt = iter(target_loader)
            
            for _ in range(min_len):
                # Source Batch (Has Labels)
                x_s, y_s, _, d_s = next(iter_src) 
                # Target Batch (No Labels)
                x_t, _, _, _ = next(iter_tgt)
                
                x_s, y_s, d_s = x_s.to(self.device), y_s.to(self.device), d_s.to(self.device)
                x_t = x_t.to(self.device)
                d_t = torch.full((x_t.size(0),), target_domain_idx, dtype=torch.long).to(self.device)
                
                optimizer.zero_grad()
                z_s = self.encoder(x_s); z_t = self.encoder(x_t)
                
                # --- LOSSES ---
                # 1. Source Replay (Task Loss)
                pred_s = self.classifier(z_s)
                l_src = self.criterion.task_loss(pred_s, y_s)
                
                # 2. Domain Adversarial (Multi-class)
                # Predict domain for Source and Target
                dom_pred_s = self.discriminator(z_s)
                dom_pred_t = self.discriminator(z_t)
                
                # Cross Entropy for Domain Classification
                l_adv = self.criterion.ce(dom_pred_s, d_s) + self.criterion.ce(dom_pred_t, d_t)
                
                # 3. Physics & Entropy (On Target)
                l_ent = self.criterion.entropy_loss(self.classifier(z_t))
                _, base_t = self.physics_head(z_t)
                l_mono = self.criterion.monotonicity_loss(base_t, prev_baseline, drift_dir)
                
                # 4. Hybrid Gating Contrastive with MEAN TEACHER
                # Use EMA models for more stable pseudo-labels
                with torch.no_grad():
                    z_t_ema = self.ema_encoder(x_t)
                    logits_t_ema = self.ema_classifier(z_t_ema)
                    probs = torch.softmax(logits_t_ema, dim=1)
                    conf, plabels = torch.max(probs, dim=1)
                    mask = torch.zeros_like(conf, dtype=torch.bool)
                    for c in range(6):
                        idx = (plabels == c).nonzero(as_tuple=True)[0]
                        if len(idx) > 0:
                            thresh = torch.quantile(conf[idx], 0.5)
                            mask[idx[conf[idx] >= torch.max(thresh, torch.tensor(0.4).to(self.device))]] = True
                
                l_cont = torch.tensor(0.0, device=self.device)
                if mask.sum() > 0:
                    z_c = torch.cat([z_s, z_t[mask]], dim=0)
                    y_c = torch.cat([y_s, plabels[mask]], dim=0)
                    l_cont = self.criterion.contrastive_loss(z_c, y_c)
                
                # 5. Cross-domain centroid alignment
                l_cross = torch.tensor(0.0, device=self.device)
                if mask.sum() > 0:
                    for c in range(6):
                        src_c = z_s[y_s == c]
                        tgt_c = z_t[mask & (plabels == c)]
                        if len(src_c) > 0 and len(tgt_c) > 0:
                            l_cross = l_cross + nn.functional.mse_loss(src_c.mean(0), tgt_c.mean(0))
                
                # 6. PROTOTYPE CONTRASTIVE LOSS (NEW - pulls features toward class prototypes)
                l_proto = torch.tensor(0.0, device=self.device)
                if self.prototype_counts.sum() > 0:
                    l_proto = self.criterion.prototype_contrastive_loss(z_s, y_s, self.prototypes)
                
                # 7. PHYSICS CONSISTENCY LOSS (NEW - same class has same physics across domains)
                l_phys = torch.tensor(0.0, device=self.device)
                if mask.sum() > 0:
                    l_phys = self.criterion.physics_consistency_loss(z_s, z_t, y_s, plabels, mask, self.physics_head)
                
                # 8. TEMPORAL ENSEMBLE LOSS (NEW - consistency with EMA predictions)
                l_te = torch.tensor(0.0, device=self.device)
                with torch.no_grad():
                    ema_logits = self.ema_classifier(self.ema_encoder(x_t))
                l_te = self.criterion.temporal_ensemble_loss(self.classifier(z_t), ema_logits)

                # TOTAL (with all advanced losses)
                loss = (0.1 * l_adv) + (1.5 * l_cont) + (0.5 * l_mono) + (0.1 * l_ent) + \
                       (0.5 * l_src) + (0.3 * l_cross) + (0.5 * l_proto) + (0.3 * l_phys) + (0.5 * l_te)
                
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                optimizer.step()
                self._update_ema()  # Update Mean Teacher
                self._update_prototypes(z_s, y_s)  # Update prototype memory
            
            if (epoch+1)%5==0: print(f"  Epoch {epoch+1}: Adv {l_adv:.3f} | Cont {l_cont:.3f} | Src {l_src:.3f}")
            
        return self._estimate_baseline(target_loader)

    # --- BRIDGE (With Domain Expansion) ---
    def update_physics_only(self, loader, prev_baseline, drift_dir):
        print("  >>> Bridging Gap: Updating Physics Baseline only...")
        # IMPORTANT: Expand discriminator to keep domain labels consistent
        self.discriminator.add_new_domain()
        self.discriminator.to(self.device)
        
        self.encoder.eval(); self.classifier.eval(); self.discriminator.eval()
        self.physics_head.train()
        for p in self.physics_head.net.parameters(): p.requires_grad = True
        optimizer = optim.Adam(self.physics_head.parameters(), lr=0.001)
        
        accum = []
        for _ in range(5):
            for x, _, _, _ in loader:
                x = x.to(self.device)
                z = self._get_z(x) # Frozen encoder
                _, base_t = self.physics_head(z)
                l_mono = self.criterion.monotonicity_loss(base_t, prev_baseline, drift_dir)
                optimizer.zero_grad(); l_mono.backward(); optimizer.step()
                accum.append(base_t.mean().item())
        return np.mean(accum)

    # --- RF Helper ---
    def train_rf(self, loader):
        print("  > Training RF on Memory...")
        self.encoder.eval()
        all_z, all_y = [], []
        with torch.no_grad():
            for x, y, _, _ in loader:
                z = self.encoder(x.to(self.device))
                all_z.append(z.cpu().numpy()); all_y.append(y.numpy())
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        rf.fit(np.concatenate(all_z), np.concatenate(all_y))
        return rf

    def generate_ensemble_pseudo_labels(self, loader, rf_model, threshold=0.75):
        self.encoder.eval(); self.classifier.eval()
        confs, preds_nn, preds_rf = [], [], []
        with torch.no_grad():
            for x, _, _, _ in loader:
                z = self.encoder(x.to(self.device))
                probs = torch.softmax(self.classifier(z), dim=1)
                conf, pred = torch.max(probs, dim=1)
                confs.append(conf.cpu().numpy())
                preds_nn.append(pred.cpu().numpy())
                preds_rf.append(rf_model.predict(z.cpu().numpy()))
        
        c = np.concatenate(confs); p_nn = np.concatenate(preds_nn); p_rf = np.concatenate(preds_rf)
        mask = (c > threshold) & (p_nn == p_rf) # Ensemble Agreement
        return p_nn, np.where(mask)[0]

    def _estimate_baseline(self, loader):
        self.encoder.eval(); self.physics_head.eval()
        accum = []
        with torch.no_grad():
            for x, _, _, _ in loader:
                z = self.encoder(x.to(self.device))
                _, base = self.physics_head(z)
                accum.append(base.mean().item())
        return np.mean(accum)

    def evaluate(self, loader):
        self.encoder.eval(); self.classifier.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y, _, _ in loader:
                pred = self.classifier(self.encoder(x.to(self.device)))
                correct += (pred.argmax(1) == y.to(self.device)).sum().item()
                total += x.size(0)
        return 100 * correct / total

    def evaluate_detailed(self, loader):
        """Evaluate with complete metrics: Accuracy, Precision, Recall, F1"""
        from src.evaluator import evaluate_detailed
        return evaluate_detailed(self.encoder, self.classifier, loader, self.device, self.num_classes)

    def save_checkpoint(self, name):
        torch.save({'enc': self.encoder.state_dict(), 'cls': self.classifier.state_dict(), 'phy': self.physics_head.state_dict()}, os.path.join(self.save_dir, name))