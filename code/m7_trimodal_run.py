#!/usr/bin/env python3
"""M7 Tri-Modal Continual Learning on MPS (Apple Silicon)."""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path

import snntorch as snn
from snntorch import surrogate
import tonic

# ── MPS/CPU compatibility helpers ────────────────────────────────────────────
def safe_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Device ───────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA")
else:
    device = torch.device('cpu')
    print("Using CPU")

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR = Path('./ICONS_M7')
DATA_DIR = BASE_DIR / 'datasets'
RESULTS_DIR = BASE_DIR / 'results'
CKPT_DIR = RESULTS_DIR / 'checkpoints'
PLOT_DIR = RESULTS_DIR / 'plots'
for d in [RESULTS_DIR, CKPT_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'max_epochs': 50,
    'patience': 12,
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
#  COLLATE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
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

def shd_collate_fn(batch):
    events, labels = zip(*batch)
    return torch.stack(events), torch.tensor(labels)

def dvs_collate_fn(batch):
    events, labels = zip(*batch)
    fixed = []
    for e in events:
        if not isinstance(e, torch.Tensor):
            e = torch.from_numpy(e).float()
        if e.shape[0] > 25:
            e = e[:25]
        elif e.shape[0] < 25:
            e = torch.cat([e, torch.zeros(25 - e.shape[0], *e.shape[1:])], dim=0)
        if e.shape[-2:] != (34, 34):
            e = F.adaptive_avg_pool2d(
                e.view(-1, e.shape[-3], e.shape[-2], e.shape[-1]),
                (34, 34)
            ).view(25, e.shape[-3], 34, 34)
        fixed.append(e)
    return torch.stack(fixed), torch.tensor(labels)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADERS
# ═══════════════════════════════════════════════════════════════════════════════
print("="*80)
print("📊 LOADING DATASETS")
print("="*80)

# N-MNIST
sensor_size = tonic.datasets.NMNIST.sensor_size
nmnist_transform = tonic.transforms.Compose([
    tonic.transforms.Denoise(filter_time=10000),
    tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
])
nmnist_train = tonic.datasets.NMNIST(save_to=str(DATA_DIR), train=True, transform=nmnist_transform)
nmnist_test  = tonic.datasets.NMNIST(save_to=str(DATA_DIR), train=False, transform=nmnist_transform)
cached_nmnist_train = tonic.DiskCachedDataset(nmnist_train, cache_path=str(DATA_DIR / 'nmnist_train_cache'))
n_val_nmnist = int(0.1 * len(cached_nmnist_train))
n_train_nmnist = len(cached_nmnist_train) - n_val_nmnist
nmnist_train_split, nmnist_val_split = random_split(
    cached_nmnist_train, [n_train_nmnist, n_val_nmnist],
    generator=torch.Generator().manual_seed(42))
train_loader_nmnist = DataLoader(nmnist_train_split, batch_size=CONFIG['batch_size'], shuffle=True,
                                  num_workers=0, collate_fn=nmnist_collate_fn)
val_loader_nmnist   = DataLoader(nmnist_val_split,   batch_size=CONFIG['batch_size'], shuffle=False,
                                  num_workers=0, collate_fn=nmnist_collate_fn)
test_loader_nmnist  = DataLoader(nmnist_test, batch_size=CONFIG['batch_size'], shuffle=False,
                                  num_workers=0, collate_fn=nmnist_collate_fn)
print(f"  N-MNIST:  train={len(nmnist_train_split)}, val={len(nmnist_val_split)}, test={len(nmnist_test)}")

# SHD
shd_train = tonic.datasets.SHD(save_to=str(DATA_DIR), train=True)
shd_test  = tonic.datasets.SHD(save_to=str(DATA_DIR), train=False)
def events_to_dense(events, label):
    time_bins, channels = 100, 700
    dense = torch.zeros(time_bins, channels)
    if len(events) > 0:
        max_time = events['t'].max() if len(events) > 0 else 1
        time_indices = (events['t'] / max_time * (time_bins - 1)).astype(int)
        channel_indices = events['x'].astype(int)
        for t, c in zip(time_indices, channel_indices):
            if 0 <= t < time_bins and 0 <= c < channels:
                dense[t, c] = 1.0
    return dense.unsqueeze(1).unsqueeze(1), label
shd_train_data = [events_to_dense(e, l) for e, l in tqdm(shd_train, desc="  SHD train", disable=False)]
shd_test_data  = [events_to_dense(e, l) for e, l in tqdm(shd_test,  desc="  SHD test", disable=False)]
n_val_shd = int(0.1 * len(shd_train_data))
shd_train_split, shd_val_split = random_split(
    shd_train_data, [len(shd_train_data) - n_val_shd, n_val_shd],
    generator=torch.Generator().manual_seed(42))
train_loader_shd = DataLoader(shd_train_split, batch_size=CONFIG['batch_size'], shuffle=True,
                               num_workers=0, collate_fn=shd_collate_fn)
val_loader_shd   = DataLoader(shd_val_split,   batch_size=CONFIG['batch_size'], shuffle=False,
                               num_workers=0, collate_fn=shd_collate_fn)
test_loader_shd  = DataLoader(shd_test_data,  batch_size=CONFIG['batch_size'], shuffle=False,
                               num_workers=0, collate_fn=shd_collate_fn)
print(f"  SHD:      train={len(shd_train_split)}, val={len(shd_val_split)}, test={len(shd_test_data)}")

# DVS-Gesture
dvs_transform = tonic.transforms.Compose([
    tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, time_window=50000),
])
dvs_train = tonic.datasets.DVSGesture(save_to=str(DATA_DIR), train=True, transform=dvs_transform)
dvs_test  = tonic.datasets.DVSGesture(save_to=str(DATA_DIR), train=False, transform=dvs_transform)
n_val_dvs = int(0.1 * len(dvs_train))
dvs_train_split, dvs_val_split = random_split(
    dvs_train, [len(dvs_train) - n_val_dvs, n_val_dvs],
    generator=torch.Generator().manual_seed(42))
train_loader_dvs = DataLoader(dvs_train_split, batch_size=CONFIG['batch_size'], shuffle=True,
                               num_workers=0, collate_fn=dvs_collate_fn)
val_loader_dvs   = DataLoader(dvs_val_split,   batch_size=CONFIG['batch_size'], shuffle=False,
                               num_workers=0, collate_fn=dvs_collate_fn)
test_loader_dvs  = DataLoader(dvs_test,  batch_size=CONFIG['batch_size'], shuffle=False,
                               num_workers=0, collate_fn=dvs_collate_fn)
print(f"  DVS:      train={len(dvs_train_split)}, val={len(dvs_val_split)}, test={len(dvs_test)}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        return -mean_log_prob.mean()

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
        self.W_r = nn.Linear(dim * 2, dim)
        self.W_z = nn.Linear(dim * 2, dim)
        self.W_h = nn.Linear(dim * 2, dim)
        self.norm_r = nn.LayerNorm(dim)
        self.norm_z = nn.LayerNorm(dim)
        self.norm_h = nn.LayerNorm(dim)
    def forward(self, x, h_prev=None):
        batch_size = x.size(0)
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.W_r.weight.size(1) // 2, device=x.device)
        concat = torch.cat([x, h_prev], dim=-1)
        r = torch.sigmoid(self.norm_r(self.W_r(concat)))
        z = torch.sigmoid(self.norm_z(self.W_z(concat)))
        h_tilde = torch.tanh(self.norm_h(self.W_h(torch.cat([x, r * h_prev], dim=-1))))
        return (1 - z) * h_prev + z * h_tilde

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
        B, T = x.size(0), x.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        spk4_rec, spk5_rec = [], []
        for t in range(T):
            xt = x[:, t]
            c1 = self.pool1(self.conv1(xt))
            s1, mem1 = self.lif1(c1, mem1)
            c2 = self.pool2(self.conv2(s1))
            s2, mem2 = self.lif2(c2, mem2)
            f = s2.view(B, -1)
            f1 = self.fc1(f)
            s3, mem3 = self.lif3(f1, mem3)
            f2 = self.fc2(s3)
            s4, mem4 = self.lif4(f2, mem4)
            spk4_rec.append(s4)
            f3 = self.fc3(s4)
            s5, mem5 = self.lif5(f3, mem5)
            spk5_rec.append(s5)
        return torch.stack(spk5_rec, dim=1).sum(dim=1), torch.stack(spk4_rec, dim=1).sum(dim=1)

class M7_ContinualAdaptiveModel(nn.Module):
    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks
        self.backbones = nn.ModuleDict()
        for name, info in tasks.items():
            if info['type'] == 'audio':
                from torch.nn import Linear, LayerNorm
                self.backbones[name] = nn.ModuleDict({
                    'fc1': nn.Linear(700, 1024),
                    'lif1': snn.Leaky(beta=CONFIG['beta'], spike_grad=spike_grad),
                    'fc2': nn.Linear(1024, 1024),
                    'lif2': snn.Leaky(beta=CONFIG['beta'], spike_grad=spike_grad),
                    'fc3': nn.Linear(1024, 512),
                    'lif3': snn.Leaky(beta=CONFIG['beta'], spike_grad=spike_grad),
                    'fc4': nn.Linear(512, info['num_classes']),
                    'lif4': snn.Leaky(beta=CONFIG['beta'], spike_grad=spike_grad),
                })
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
        bb = self.backbones[modality]
        if isinstance(bb, SNN_Backbone):
            _, features = bb(x)
        else:
            B, T = x.size(0), x.size(1)
            mem1 = bb['lif1'].init_leaky()
            mem2 = bb['lif2'].init_leaky()
            mem3 = bb['lif3'].init_leaky()
            feats = []
            for t in range(T):
                xt = x[:, t].view(B, -1)
                s1, mem1 = bb['lif1'](bb['fc1'](xt), mem1)
                s2, mem2 = bb['lif2'](bb['fc2'](s1), mem2)
                s3, mem3 = bb['lif3'](bb['fc3'](s2), mem3)
                feats.append(s3)
            features = torch.stack(feats, dim=1).sum(dim=1)
        return features.unsqueeze(1)

    def forward(self, x, modality, task_name):
        bb = self.backbones[modality]
        if isinstance(bb, SNN_Backbone):
            _, features = bb(x)
        else:
            B, T = x.size(0), x.size(1)
            mem1 = bb['lif1'].init_leaky()
            mem2 = bb['lif2'].init_leaky()
            mem3 = bb['lif3'].init_leaky()
            feats = []
            for t in range(T):
                xt = x[:, t].view(B, -1)
                s1, mem1 = bb['lif1'](bb['fc1'](xt), mem1)
                s2, mem2 = bb['lif2'](bb['fc2'](s1), mem2)
                s3, mem3 = bb['lif3'](bb['fc3'](s2), mem3)
                feats.append(s3)
            features = torch.stack(feats, dim=1).sum(dim=1)
        gate = self.adaptive_gate(features)
        hop = self.hopfield(features)
        hgrn = self.hgrn(features)
        combined = gate[:, 0:1] * hop + gate[:, 1:2] * hgrn
        return self.heads[task_name](combined), gate, features

    def forward_from_embeddings(self, embs, task_name):
        gate = self.adaptive_gate(embs)
        hop = self.hopfield(embs)
        hgrn = self.hgrn(embs)
        combined = gate[:, 0:1] * hop + gate[:, 1:2] * hgrn
        return self.heads[task_name](combined)

class EmbeddingReplayBuffer:
    def __init__(self, feature_dim=512, samples_per_class=50, device=None):
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
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
                data = data.to(next(model.parameters()).device)
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
        return torch.stack(embs).to(self.device), torch.tensor(lbls, dtype=torch.long, device=self.device)

    def stats(self):
        return {'total': sum(len(v) for v in self._store.values()),
                'num_classes': len(self._store)}

class ContinualLearner:
    def __init__(self, model, buffer, device=None, replay_batch=32, replay_weight=0.3):
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
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
            out, _, _ = self.model(data, modality, task_name)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return 100.0 * correct / total if total > 0 else 0.0

    def train_task(self, task_name, modality, train_loader, val_loader, test_loader,
                   num_epochs=50, patience=12, lr=1e-3, use_replay=False, replay_task_ids=None,
                   use_scl=True, scl_weight=0.1, step_id=0):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        ce = nn.CrossEntropyLoss()
        scl = SupervisedContrastiveLoss(temperature=CONFIG['contrastive_temperature']).to(self.device)

        best_val = 0
        patience_counter = 0
        history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'gate_hop': [], 'gate_hgrn': []}

        for epoch in range(num_epochs):
            self.model.train()
            train_correct = train_total = 0
            train_loss_sum = 0.0
            gate_hop_sum = gate_hgrn_sum = gate_count = 0

            total_batches = len(train_loader)
            print_every = max(1, total_batches // 10)
            for i, (data, labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits, gate, features = self.model(data, modality, task_name)
                loss = ce(logits, labels)

                if use_scl:
                    loss = loss + scl_weight * scl(features, labels)

                if use_replay and self.buffer.stats()['total'] > 0:
                    replay_embs, replay_labels = self.buffer.sample(self.replay_batch, replay_task_ids)
                    if replay_embs is not None:
                        replay_logits = self.model.forward_from_embeddings(replay_embs, task_name)
                        loss = loss + self.replay_weight * ce(replay_logits, replay_labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG['gradient_clip'])
                optimizer.step()

                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += labels.size(0)
                train_loss_sum += loss.item()
                gate_hop_sum += gate[:, 0].sum().item()
                gate_hgrn_sum += gate[:, 1].sum().item()
                gate_count += gate.size(0)
                if (i + 1) % print_every == 0 or i == total_batches - 1:
                    print(f"    batch {i+1:4d}/{total_batches} | loss={train_loss_sum/(i+1):.3f} | acc={100*train_correct/train_total:.1f}%")

            scheduler.step()
            train_acc = 100.0 * train_correct / train_total
            val_acc = self.evaluate(val_loader, modality, task_name)

            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss_sum / len(train_loader))
            history['gate_hop'].append(gate_hop_sum / gate_count)
            history['gate_hgrn'].append(gate_hgrn_sum / gate_count)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [S{step_id}] Epoch {epoch+1:02d}/{num_epochs} | "
                      f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
                      f"hop={history['gate_hop'][-1]:.3f} hgrn={history['gate_hgrn'][-1]:.3f}")

            # Save best checkpoint
            if val_acc > best_val:
                best_val = val_acc
                patience_counter = 0
                ckpt_path = CKPT_DIR / f"best_{task_name}_{modality}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                }, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Save final checkpoint (caller handles condition suffix via rename after return)

        test_acc = self.evaluate(test_loader, modality, task_name)
        return {'test_acc': test_acc, 'best_val': best_val, 'history': history}

# ═══════════════════════════════════════════════════════════════════════════════
#  TRI-MODAL CL EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════
TASKS = {
    'nmnist': {'num_classes': 10, 'type': 'visual'},
    'shd':    {'num_classes': 20, 'type': 'audio'},
    'dvs':    {'num_classes': 11, 'type': 'visual_motion'},
}

SEEDS = [123, 456]  # Run additional seeds for statistical significance
CONDITIONS = ['no_replay', 'replay']

def run_one_seed(seed, condition):
    use_replay = (condition == 'replay')
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n{'='*80}")
    print(f"  SEED {seed} | {condition.upper()}")
    print(f"{'='*80}")

    model = M7_ContinualAdaptiveModel(tasks=TASKS).to(device)
    buffer = EmbeddingReplayBuffer(feature_dim=512, samples_per_class=50, device=str(device))
    trainer = ContinualLearner(model=model, buffer=buffer, device=str(device),
                               replay_batch=32, replay_weight=0.3)

    # ── T1: N-MNIST ──
    print("\n📌 STEP 1: N-MNIST (visual)")
    ckpt_t1 = CKPT_DIR / f'final_nmnist_nmnist_seed{seed}_{condition}.pt'
    if ckpt_t1.exists():
        print(f"   ⏩ Skipping T1 training (checkpoint found)")
        model.load_state_dict(torch.load(ckpt_t1, map_location=str(device))['model_state_dict'])
    else:
        r1 = trainer.train_task('nmnist', 'nmnist', train_loader_nmnist, val_loader_nmnist, test_loader_nmnist,
                                num_epochs=15, patience=5, lr=1e-3, use_replay=False, use_scl=True, scl_weight=0.1, step_id=1)
        torch.save({'model_state_dict': model.state_dict()}, CKPT_DIR / f'final_nmnist_nmnist_seed{seed}_{condition}.pt')
    acc_t1 = trainer.evaluate(test_loader_nmnist, 'nmnist', 'nmnist')
    print(f"   T1 Test: {acc_t1:.2f}%")

    # Save incremental result after T1
    _save_incremental(seed, condition, 'T1_after_T1', acc_t1)

    # Populate buffer with T1 embeddings
    if use_replay:
        buffer.populate(model, train_loader_nmnist, 'nmnist', task_id=1, label_offset=0)
        print(f"   Buffer T1: {buffer.stats()['total']} embeddings")

    # ── T2: SHD ──
    print("\n📌 STEP 2: SHD (audio)")
    # Zero-shot T2 before training (for FWT)
    acc_t2_zero_shot = trainer.evaluate(test_loader_shd, 'shd', 'shd')
    print(f"   T2 zero-shot: {acc_t2_zero_shot:.2f}%")

    ckpt_t2 = CKPT_DIR / f'final_shd_shd_seed{seed}_{condition}.pt'
    if ckpt_t2.exists():
        print(f"   ⏩ Skipping T2 training (checkpoint found)")
        model.load_state_dict(torch.load(ckpt_t2, map_location=str(device))['model_state_dict'])
    else:
        r2 = trainer.train_task('shd', 'shd', train_loader_shd, val_loader_shd, test_loader_shd,
                                num_epochs=15, patience=5, lr=1e-3,
                                use_replay=use_replay, replay_task_ids=[1] if use_replay else None,
                                use_scl=True, scl_weight=0.1, step_id=2)
        torch.save({'model_state_dict': model.state_dict()}, CKPT_DIR / f'final_shd_shd_seed{seed}_{condition}.pt')
    acc_t2_after_t2 = trainer.evaluate(test_loader_shd, 'shd', 'shd')
    print(f"   T2 Test: {acc_t2_after_t2:.2f}%")

    # Save incremental result after T2
    _save_incremental(seed, condition, 'T2_after_T2', acc_t2_after_t2)

    # Re-evaluate T1 after T2
    acc_t1_after_t2 = trainer.evaluate(test_loader_nmnist, 'nmnist', 'nmnist')
    print(f"   T1 after T2: {acc_t1_after_t2:.2f}%")

    if use_replay:
        buffer.populate(model, train_loader_shd, 'shd', task_id=2, label_offset=0)
        print(f"   Buffer T2: {buffer.stats()['total']} embeddings")

    # ── T3: DVS-Gesture ──
    print("\n📌 STEP 3: DVS-Gesture (visual-motion)")
    ckpt_t3 = CKPT_DIR / f'final_dvs_dvs_seed{seed}_{condition}.pt'
    if ckpt_t3.exists():
        print(f"   ⏩ Skipping T3 training (checkpoint found)")
        model.load_state_dict(torch.load(ckpt_t3, map_location=str(device))['model_state_dict'])
    else:
        r3 = trainer.train_task('dvs', 'dvs', train_loader_dvs, val_loader_dvs, test_loader_dvs,
                                num_epochs=15, patience=5, lr=1e-3,
                                use_replay=use_replay, replay_task_ids=[1, 2] if use_replay else None,
                                use_scl=True, scl_weight=0.1, step_id=3)
        torch.save({'model_state_dict': model.state_dict()}, CKPT_DIR / f'final_dvs_dvs_seed{seed}_{condition}.pt')
    acc_t3 = trainer.evaluate(test_loader_dvs, 'dvs', 'dvs')
    print(f"   T3 Test: {acc_t3:.2f}%")

    # Save incremental result after T3
    _save_incremental(seed, condition, 'T3_after_T3', acc_t3)

    # Re-evaluate T1 and T2 after T3
    acc_t1_after_t3 = trainer.evaluate(test_loader_nmnist, 'nmnist', 'nmnist')
    acc_t2_after_t3 = trainer.evaluate(test_loader_shd, 'shd', 'shd')
    print(f"   T1 after T3: {acc_t1_after_t3:.2f}%")
    print(f"   T2 after T3: {acc_t2_after_t3:.2f}%")

    # Metrics
    forgetting_t1 = acc_t1 - acc_t1_after_t2
    forgetting_t1_t3 = acc_t1_after_t2 - acc_t1_after_t3
    forgetting_t2_t3 = acc_t2_after_t2 - acc_t2_after_t3
    fwt = acc_t2_after_t2 - acc_t2_zero_shot

    result = {
        'seed': seed,
        'condition': condition,
        'T1_after_T1': acc_t1,
        'T2_after_T2': acc_t2_after_t2,
        'T3_after_T3': acc_t3,
        'T1_after_T2': acc_t1_after_t2,
        'T1_after_T3': acc_t1_after_t3,
        'T2_after_T3': acc_t2_after_t3,
        'forgetting_T1_T2': forgetting_t1,
        'forgetting_T1_T3': forgetting_t1_t3,
        'forgetting_T2_T3': forgetting_t2_t3,
        'fwt': fwt,
    }

    # Save
    csv_path = RESULTS_DIR / 'cl_trimodal_results.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    else:
        df = pd.DataFrame([result])
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved to {csv_path}")

    return result

# ═══════════════════════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════════════════════
import pandas as pd

def _save_incremental(seed, condition, metric_name, metric_value):
    """Append incremental result to CSV after each task."""
    csv_path = RESULTS_DIR / 'cl_trimodal_results.csv'
    row = {'seed': seed, 'condition': condition, 'step': metric_name, metric_name: metric_value}
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)
    print(f"   💾 Incremental save: {metric_name}={metric_value:.2f}%")

print("\n" + "="*80)
print("🚀 M7 TRI-MODAL CONTINUAL LEARNING")
print("="*80)
print(f"Device: {device}")
print(f"Seeds: {SEEDS}")
print(f"Conditions: {CONDITIONS}")
print()

all_results = []
for seed in SEEDS:
    for condition in CONDITIONS:
        result = run_one_seed(seed, condition)
        all_results.append(result)

# Summary
print("\n" + "="*80)
print("📊 RESULTS SUMMARY")
print("="*80)
df = pd.DataFrame(all_results)
print(df.to_string(index=False))
print("\n✅ Tri-modal CL complete!")
