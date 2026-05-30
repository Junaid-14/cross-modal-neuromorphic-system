#!/usr/bin/env python3
"""N-MNIST Smoke Test on MPS — validates M7 model trains correctly on Apple Silicon."""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import snntorch as snn
from snntorch import surrogate, utils
import tonic
from tonic import datasets, transforms

# ── MPS/CPU compatibility helpers ────────────────────────────────────────────
def safe_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # MPS has no empty_cache API; OS manages unified memory

# ── Device ───────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"✅ Device: {device}")
if torch.backends.mps.is_available():
    print("   GPU: Apple Silicon MPS")
    print("   Memory: Shared unified memory (OS managed)")
elif torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ── Directories ───────────────────────────────────────────────────────────────
from pathlib import Path
BASE_DIR = Path('./ICONS_M7')
DATA_DIR = BASE_DIR / 'datasets'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'max_epochs': 30,
    'patience': 5,
    'gradient_clip': 1.0,
    'beta': 0.9,
    'dropout': 0.2,
    'hidden_dim': 512,
    'num_patterns': 100,
    'num_gru_layers': 2,
    'num_workers': 0,
    'time_steps': 25,
    'pin_memory': True,
    'use_contrastive': True,
    'contrastive_temperature': 0.07,
    'contrastive_weight': 0.1,
    'seed': 42,
    'device': device,
}
spike_grad = surrogate.atan()

# ═══════════════════════════════════════════════════════════════════════════════
#  N-MNIST LOADER (same as notebook Cell 7)
# ═══════════════════════════════════════════════════════════════════════════════
print("="*80)
print("📊 N-MNIST DATASET LOADING")
print("="*80)

sensor_size = tonic.datasets.NMNIST.sensor_size

def nmnist_collate_fn(batch):
    events, labels = zip(*batch)
    fixed = []
    for e in events:
        if not isinstance(e, torch.Tensor):
            e = torch.from_numpy(e).float()
        if e.shape[0] > 25:
            e = e[:25]
        elif e.shape[0] < 25:
            e = torch.cat([e, torch.zeros(25 - e.shape[0], *e.shape[1:])], dim=0)
        fixed.append(e)
    return torch.stack(fixed), torch.tensor(labels)

transform = tonic.transforms.Compose([
    tonic.transforms.Denoise(filter_time=10000),
    tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
])

train_dataset = tonic.datasets.NMNIST(save_to=str(DATA_DIR), train=True, transform=transform)
test_dataset  = tonic.datasets.NMNIST(save_to=str(DATA_DIR), train=False, transform=transform)

# Use a subset for quick MPS validation (full dataset would take hours)
QUICK_SUBSET = 5000
indices = torch.randperm(len(train_dataset))[:QUICK_SUBSET]
train_subset = torch.utils.data.Subset(train_dataset, indices)

n_val = int(0.1 * len(train_subset))
n_train = len(train_subset) - n_val
train_ds, val_ds = random_split(train_subset, [n_train, n_val],
                                 generator=torch.Generator().manual_seed(42))

train_loader_nmnist = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                                  num_workers=0, collate_fn=nmnist_collate_fn)
val_loader_nmnist   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'], shuffle=False,
                                  num_workers=0, collate_fn=nmnist_collate_fn)
test_loader_nmnist  = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                                  num_workers=0, collate_fn=nmnist_collate_fn)

print(f"   Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_dataset):,}")

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS (extracted from notebook)
# ═══════════════════════════════════════════════════════════════════════════════

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss

class ModernHopfieldLayer(nn.Module):
    def __init__(self, dim=512, num_patterns=256, temperature=1.0):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_patterns, dim) * 0.02)
        self.temperature = temperature
        self.scale = np.sqrt(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn = torch.matmul(x, self.memory.T) / self.temperature / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, self.memory)
        return self.norm(x + out)

class ImprovedHGRNGate(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.W_r = nn.Linear(dim * 2, dim)
        self.W_z = nn.Linear(dim * 2, dim)
        self.W_h = nn.Linear(dim * 2, dim)
        self.norm_r = nn.LayerNorm(dim)
        self.norm_z = nn.LayerNorm(dim)
        self.norm_h = nn.LayerNorm(dim)

    def forward(self, x, h_prev=None):
        batch_size = x.size(0)
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.dim, device=x.device)
        concat = torch.cat([x, h_prev], dim=-1)
        r = torch.sigmoid(self.norm_r(self.W_r(concat)))
        z = torch.sigmoid(self.norm_z(self.W_z(concat)))
        h_tilde = torch.tanh(self.norm_h(self.W_h(torch.cat([x, r * h_prev], dim=-1))))
        h = (1 - z) * h_prev + z * h_tilde
        return h

class SNN_Backbone(nn.Module):
    def __init__(self, input_channels=2, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=CONFIG['beta'], spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=CONFIG['beta'], spike_grad=spike_grad)
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.lif3 = snn.Leaky(beta=CONFIG['beta'], spike_grad=spike_grad)
        self.fc2 = nn.Linear(1024, 512)
        self.lif4 = snn.Leaky(beta=CONFIG['beta'], spike_grad=spike_grad)
        self.fc3 = nn.Linear(512, num_classes)
        self.lif5 = snn.Leaky(beta=CONFIG['beta'], spike_grad=spike_grad)

    def forward(self, x):
        batch_size, time_steps = x.size(0), x.size(1)
        spk1_rec, spk2_rec, spk3_rec, spk4_rec, spk5_rec = [], [], [], [], []
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        for t in range(time_steps):
            xt = x[:, t, :, :, :]
            c1 = self.pool1(self.conv1(xt))
            spk1, mem1 = self.lif1(c1, mem1)
            spk1_rec.append(spk1)
            c2 = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(c2, mem2)
            spk2_rec.append(spk2)
            flat = spk2.view(batch_size, -1)
            f1 = self.fc1(flat)
            spk3, mem3 = self.lif3(f1, mem3)
            spk3_rec.append(spk3)
            f2 = self.fc2(spk3)
            spk4, mem4 = self.lif4(f2, mem4)
            spk4_rec.append(spk4)
            f3 = self.fc3(spk4)
            spk5, mem5 = self.lif5(f3, mem5)
            spk5_rec.append(spk5)

        spk_sum = torch.stack(spk5_rec, dim=1).sum(dim=1)
        features = torch.stack(spk4_rec, dim=1).sum(dim=1)
        return spk_sum, features

class M7_ContinualAdaptiveModel(nn.Module):
    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks
        self.backbones = nn.ModuleDict()
        for name, info in tasks.items():
            if info['type'] == 'audio':
                continue  # not needed for N-MNIST smoke test
            else:
                self.backbones[name] = SNN_Backbone(input_channels=2, num_classes=info['num_classes'])
        self.hopfield = ModernHopfieldLayer(dim=512, num_patterns=CONFIG['num_patterns'])
        self.hgrn = ImprovedHGRNGate(dim=512)
        self.adaptive_gate = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 2), nn.Softmax(dim=-1)
        )
        self.heads = nn.ModuleDict()
        for name, info in tasks.items():
            self.heads[name] = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, info['num_classes']))

    def _get_features(self, x, modality):
        backbone = self.backbones[modality] if modality in self.backbones else self.backbones[list(self.backbones.keys())[0]]
        _, features = backbone(x)
        return features.unsqueeze(1)

    def forward(self, x, modality, task_name):
        backbone = self.backbones[modality] if modality in self.backbones else self.backbones[list(self.backbones.keys())[0]]
        _, features = backbone(x)
        gate_weights = self.adaptive_gate(features)
        w_hop = gate_weights[:, 0:1]
        w_hgrn = gate_weights[:, 1:2]
        hop_out = self.hopfield(features)
        hgrn_out = self.hgrn(features)
        combined = w_hop * hop_out + w_hgrn * hgrn_out
        logits = self.heads[task_name](combined)
        return logits, gate_weights, features

    def forward_from_embeddings(self, embs, task_name):
        gate_weights = self.adaptive_gate(embs)
        w_hop = gate_weights[:, 0:1]
        w_hgrn = gate_weights[:, 1:2]
        hop_out = self.hopfield(embs)
        hgrn_out = self.hgrn(embs)
        combined = w_hop * hop_out + w_hgrn * hgrn_out
        return self.heads[task_name](combined)

class EmbeddingReplayBuffer:
    def __init__(self, feature_dim=512, samples_per_class=50, device=None):
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.feature_dim = feature_dim
        self.samples_per_class = samples_per_class
        self.device = device
        self._store = {}
        self._task_labels = {}

    def populate(self, model, loader, modality, task_id, label_offset=0):
        model.eval()
        class_samples = {}
        with torch.no_grad():
            for data, labels in loader:
                data = data.to(model.backbones[list(model.backbones.keys())[0]].conv1.weight.device)
                labels = labels.to(data.device)
                feats = model._get_features(data, modality).squeeze(1)
                for f, lbl in zip(feats, labels):
                    c = int(lbl.item()) + label_offset
                    if c not in class_samples:
                        class_samples[c] = []
                    class_samples[c].append(f.cpu())
        for c, samples in class_samples.items():
            if len(samples) > self.samples_per_class:
                indices = torch.randperm(len(samples))[:self.samples_per_class]
                samples = [samples[i] for i in indices]
            self._store[c] = torch.stack(samples)
            self._task_labels[c] = task_id

    def sample(self, batch_size, task_ids=None):
        if task_ids is None:
            task_ids = list(self._store.keys())
        available = [c for c in task_ids if c in self._store]
        if not available:
            return None, None
        classes = np.random.choice(available, size=min(batch_size, len(available)), replace=False)
        embs, lbls = [], []
        for c in classes:
            idx = np.random.randint(len(self._store[c]))
            embs.append(self._store[c][idx])
            lbls.append(c)
        embs = torch.stack(embs).to(self.device)
        lbls = torch.tensor(lbls, dtype=torch.long, device=self.device)
        return embs, lbls

    def stats(self):
        return {'total': sum(len(v) for v in self._store.values()),
                'num_classes': len(self._store),
                'by_task': {c: len(v) for c, v in self._store.items()}}

class ContinualLearner:
    def __init__(self, model, buffer, device=None, replay_batch=32, replay_weight=0.3):
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(device)
        self.buffer = buffer
        self.device = device
        self.replay_batch = replay_batch
        self.replay_weight = replay_weight
        self.acc_matrix = {}
        self.history = {}

    @torch.no_grad()
    def evaluate(self, loader, modality, task_name):
        self.model.eval()
        correct = total = 0
        for data, labels in loader:
            data, labels = data.to(self.device), labels.to(self.device)
            spk, _, _ = self.model(data, modality, task_name)
            correct += (spk.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return 100.0 * correct / total if total > 0 else 0.0

    def train_task(self, task_name, modality, train_loader, val_loader, test_loader,
                   num_epochs=50, patience=12, lr=1e-3, use_replay=False, replay_task_ids=None,
                   use_scl=True, scl_weight=0.1, step_id=0):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss()
        scl_criterion = SupervisedContrastiveLoss(temperature=CONFIG['contrastive_temperature']).to(self.device)

        best_val = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_acc': [], 'train_acc': [], 'gate_hop': [], 'gate_hgrn': []}

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = train_correct = train_total = 0
            gate_hop_sum = gate_hgrn_sum = gate_count = 0

            for i, (data, labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits, gate, features = self.model(data, modality, task_name)
                loss = criterion(logits, labels)

                if use_scl:
                    scl_loss = scl_criterion(features, labels)
                    loss = loss + scl_weight * scl_loss

                if use_replay and self.buffer.stats()['total'] > 0:
                    replay_embs, replay_labels = self.buffer.sample(self.replay_batch, replay_task_ids)
                    if replay_embs is not None:
                        replay_logits = self.model.forward_from_embeddings(replay_embs, task_name)
                        replay_loss = criterion(replay_logits, replay_labels)
                        loss = loss + self.replay_weight * replay_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG['gradient_clip'])
                optimizer.step()

                train_loss += loss.item()
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += labels.size(0)
                gate_hop_sum += gate[:, 0].sum().item()
                gate_hgrn_sum += gate[:, 1].sum().item()
                gate_count += gate.size(0)
                if i % 200 == 0 and i > 0:
                    print(f"    batch {i}/{len(train_loader)} | loss: {loss.item():.4f}")

            scheduler.step()
            train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
            val_acc = self.evaluate(val_loader, modality, task_name)

            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['gate_hop'].append(gate_hop_sum / gate_count if gate_count > 0 else 0)
            history['gate_hgrn'].append(gate_hgrn_sum / gate_count if gate_count > 0 else 0)

            print(f"  Epoch {epoch+1:02d}/{num_epochs} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Loss: {history['train_loss'][-1]:.4f}")

            if val_acc > best_val:
                best_val = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        test_acc = self.evaluate(test_loader, modality, task_name)
        return {'test_acc': test_acc, 'best_val': best_val, 'history': history}

# ═══════════════════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════
print("="*80)
print("🧪 N-MNIST SINGLE-TASK SMOKE TEST (MPS)")
print("="*80)

torch.manual_seed(42)
np.random.seed(42)

TASKS_SMOKE = {'nmnist': {'num_classes': 10, 'type': 'visual'}}

model_smoke = M7_ContinualAdaptiveModel(tasks=TASKS_SMOKE).to(device)
buffer_smoke = EmbeddingReplayBuffer(feature_dim=512, samples_per_class=50, device=str(device))
trainer = ContinualLearner(model=model_smoke, buffer=buffer_smoke, device=str(device),
                           replay_batch=32, replay_weight=0.3)

print("   Quick subset: 5,000 samples | Epochs: 10 (for fast MPS validation)\n")
result_smoke = trainer.train_task(
    'nmnist', 'nmnist',
    train_loader_nmnist, val_loader_nmnist, test_loader_nmnist,
    num_epochs=10, patience=5, lr=1e-3,
    use_replay=False, use_scl=True, scl_weight=0.1,
    step_id=0)

nmnist_test_acc = result_smoke['test_acc']
nmnist_best_val = result_smoke['best_val']

print("\n" + "="*80)
print("📊 N-MNIST SMOKE TEST RESULTS")
print("="*80)
print(f"   N-MNIST Test Accuracy: {nmnist_test_acc:.2f}%")
print(f"   N-MNIST Best Val:      {nmnist_best_val:.2f}%")
print(f"   Classes:               10 (chance = {100/10:.1f}%)")

# Shape sanity checks
xb, yb = next(iter(test_loader_nmnist))
xb = xb.to(device)
logits, gate, features = model_smoke(xb, 'nmnist', 'nmnist')
print(f"\n🧪 Tensor shape checks:")
print(f"   Input:    {xb.shape}   (expect [B,25,2,34,34])")
print(f"   Logits:   {logits.shape}   (expect [B,10])")
print(f"   Gate:     {gate.shape}     (expect [B,2])")
print(f"   Features: {features.shape} (expect [B,512])")

assert logits.shape == (xb.shape[0], 10), f"Logits shape mismatch: {logits.shape}"
assert gate.shape == (xb.shape[0], 2), f"Gate shape mismatch: {gate.shape}"
assert features.shape == (xb.shape[0], 512), f"Features shape mismatch: {features.shape}"

CHANCE_NMNIST = 100.0 / 10.0
if nmnist_test_acc <= CHANCE_NMNIST + 1.0:
    print(f"\n🚨 ALERT: N-MNIST accuracy {nmnist_test_acc:.2f}% is at or below chance ({CHANCE_NMNIST:.1f}%)")
    raise RuntimeError("N-MNIST single-task accuracy below chance — aborting.")
elif nmnist_test_acc < 80.0:
    print(f"\n⚠️  WARNING: N-MNIST accuracy {nmnist_test_acc:.2f}% < 80% target")
else:
    print(f"\n✅ N-MNIST smoke test PASSED: {nmnist_test_acc:.2f}% >> chance ({CHANCE_NMNIST:.1f}%)")

del model_smoke, buffer_smoke, trainer
safe_empty_cache()
print("="*80 + "\n")
