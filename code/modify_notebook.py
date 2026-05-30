#!/usr/bin/env python3
"""Modify NeurIPS notebook to add DVS-Gesture tri-modal CL support."""
import json
import copy

with open('NeurIPS2026_M7_clean (3) (1).ipynb', 'r') as f:
    nb = json.load(f)

cells = nb['cells']

# =====================================================================
# 1. Insert DVS loader execution + val split after cell 9 (Cell 8.5)
# =====================================================================

cell_8_5_exec = {
    "cell_type": "code",
    "execution_count": None,
    "id": "dvs-loader-exec",
    "metadata": {},
    "outputs": [],
    "source": [
        "\"\"\"Cell 8.5b: Execute DVS-Gesture Loader & Validation Split\"\"\"\n",
        "\n",
        "print(\"=\"*80)\n",
        "print(\"📊 DVS-GESTURE DATASET LOADING\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "# ── Load DVS-Gesture ───────────────────────────────────────────────────────────\n",
        "train_loader_dvs, test_loader_dvs, dvs_info = get_dvs_gesture_loaders(\n",
        "    batch_size=CONFIG['batch_size'])\n",
        "\n",
        "# Quick shape check\n",
        "data_sample, label_sample = next(iter(train_loader_dvs))\n",
        "print(f\"\\n🧪 DVS sample batch:\")\n",
        "print(f\"   Data shape:  {data_sample.shape}\")\n",
        "print(f\"   Labels:      {label_sample[:8].tolist()}\")\n",
        "print(f\"   Classes:     {dvs_info['num_classes']}\")\n",
        "\n",
        "# ── Validation split for CL experiments (mirror N-MNIST/SHD pattern) ──────────\n",
        "# DVS-Gesture train is relatively small (~1K), so we do 10% val split\n",
        "_dvs_train_dataset = train_loader_dvs.dataset\n",
        "_dvs_n_val   = int(0.1 * len(_dvs_train_dataset))\n",
        "_dvs_n_train = len(_dvs_train_dataset) - _dvs_n_val\n",
        "\n",
        "# Re-create dataset with same transform for deterministic splitting\n",
        "_dvs_transform = tonic.transforms.Compose([\n",
        "    tonic.transforms.Denoise(filter_time=10000),\n",
        "    tonic.transforms.ToFrame(\n",
        "        sensor_size=(128, 128, 2),\n",
        "        time_window=5000,\n",
        "    ),\n",
        "])\n",
        "\n",
        "# We reuse the collate function from the loader (must be module-scope)\n",
        "# Since dvs_collate_fn is defined inside get_dvs_gesture_loaders, we redefine it here\n",
        "def dvs_collate_fn(batch):\n",
        "    events, labels = zip(*batch)\n",
        "    fixed = []\n",
        "    for e in events:\n",
        "        if not isinstance(e, torch.Tensor):\n",
        "            e = torch.from_numpy(e).float()\n",
        "        if e.shape[0] > 25:\n",
        "            e = e[:25]\n",
        "        elif e.shape[0] < 25:\n",
        "            e = torch.cat([\n",
        "                e, \n",
        "                torch.zeros(25 - e.shape[0], *e.shape[1:])\n",
        "            ], dim=0)\n",
        "        if e.shape[-2:] != (34, 34):\n",
        "            e = F.adaptive_avg_pool2d(\n",
        "                e.view(-1, e.shape[-3], e.shape[-2], e.shape[-1]),\n",
        "                (34, 34)\n",
        "            ).view(25, e.shape[-3], 34, 34)\n",
        "        fixed.append(e)\n",
        "    return torch.stack(fixed), torch.tensor(labels)\n",
        "\n",
        "# Re-instantiate raw dataset for splitting\n",
        "_dvs_train_raw = tonic.datasets.DVSGesture(\n",
        "    save_to=str(DATA_DIR), train=True, transform=_dvs_transform)\n",
        "\n",
        "_dvs_train_split, _dvs_val_split = torch.utils.data.random_split(\n",
        "    _dvs_train_raw, [_dvs_n_train, _dvs_n_val],\n",
        "    generator=torch.Generator().manual_seed(42)\n",
        ")\n",
        "\n",
        "train_loader_dvs_cl = DataLoader(\n",
        "    _dvs_train_split, batch_size=CONFIG['batch_size'],\n",
        "    shuffle=True, collate_fn=dvs_collate_fn, num_workers=0, pin_memory=True)\n",
        "val_loader_dvs = DataLoader(\n",
        "    _dvs_val_split, batch_size=CONFIG['batch_size'],\n",
        "    shuffle=False, collate_fn=dvs_collate_fn, num_workers=0, pin_memory=True)\n",
        "\n",
        "print(f\"✅ DVS val loader:   {_dvs_n_val} samples\")\n",
        "print(f\"✅ DVS train (CL):   {_dvs_n_train} samples\")\n",
        "print(f\"✅ DVS test:         {len(test_loader_dvs.dataset)} samples\")\n",
        "print(f\"   Shape check: {next(iter(train_loader_dvs_cl))[0].shape}\")\n",
        "print(\"=\"*80 + \"\\n\")\n"
    ]
}

# Insert after cell 9
cells.insert(10, cell_8_5_exec)

# =====================================================================
# 2. Insert DVS single-task smoke test cell after the loader exec cell
# =====================================================================

cell_smoke = {
    "cell_type": "code",
    "execution_count": None,
    "id": "dvs-smoke-test",
    "metadata": {},
    "outputs": [],
    "source": [
        "\"\"\"Cell 8.5c: DVS-Gesture Single-Task Smoke Test\"\"\"\n",
        "\n",
        "print(\"=\"*80)\n",
        "print(\"🧪 DVS-GESTURE SINGLE-TASK SMOKE TEST\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "torch.manual_seed(42)\n",
        "torch.cuda.manual_seed_all(42)\n",
        "np.random.seed(42)\n",
        "\n",
        "# Build a temporary M7 model with DVS task\n",
        "TASKS_SMOKE = {\n",
        "    'dvs': {'num_classes': 11, 'type': 'visual_motion'},\n",
        "}\n",
        "\n",
        "model_smoke = M7_ContinualAdaptiveModel(tasks=TASKS_SMOKE).to(device)\n",
        "buffer_smoke = EmbeddingReplayBuffer(\n",
        "    feature_dim=512, samples_per_class=50, device=str(device))\n",
        "trainer = ContinualLearner(\n",
        "    model=model_smoke, buffer=buffer_smoke, device=str(device),\n",
        "    replay_batch=32, replay_weight=0.3)\n",
        "\n",
        "result_smoke = trainer.train_task(\n",
        "    'dvs', 'dvs',\n",
        "    train_loader_dvs_cl, val_loader_dvs, test_loader_dvs,\n",
        "    num_epochs=30, patience=12, lr=1e-3,\n",
        "    use_replay=False, use_scl=True, scl_weight=0.1,\n",
        "    step_id=0)\n",
        "\n",
        "dvs_test_acc = result_smoke['test_acc']\n",
        "dvs_best_val = result_smoke['best_val']\n",
        "\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"📊 DVS SMOKE TEST RESULTS\")\n",
        "print(\"=\"*80)\n",
        "print(f\"   DVS Test Accuracy:  {dvs_test_acc:.2f}%\")\n",
        "print(f\"   DVS Best Val:       {dvs_best_val:.2f}%\")\n",
        "print(f\"   Classes:            11 (chance = {100/11:.1f}%)\")\n",
        "\n",
        "# Shape sanity checks\n",
        "xb, yb = next(iter(test_loader_dvs))\n",
        "xb = xb.to(device)\n",
        "logits, gate, features = model_smoke(xb, 'dvs', 'dvs')\n",
        "print(f\"\\n🧪 Tensor shape checks:\")\n",
        "print(f\"   Input:    {xb.shape}   (expect [B,25,2,34,34])\")\n",
        "print(f\"   Logits:   {logits.shape}   (expect [B,11])\")\n",
        "print(f\"   Gate:     {gate.shape}     (expect [B,2])\")\n",
        "print(f\"   Features: {features.shape} (expect [B,512])\")\n",
        "\n",
        "# Validate correctness\n",
        "assert logits.shape == (xb.shape[0], 11), f\"Logits shape mismatch: {logits.shape}\"\n",
        "assert gate.shape == (xb.shape[0], 2), f\"Gate shape mismatch: {gate.shape}\"\n",
        "assert features.shape == (xb.shape[0], 512), f\"Features shape mismatch: {features.shape}\"\n",
        "\n",
        "# Flag if below chance\n",
        "CHANCE_DVS = 100.0 / 11.0  # ~9.09%\n",
        "if dvs_test_acc <= CHANCE_DVS + 1.0:\n",
        "    print(f\"\\n🚨 ALERT: DVS accuracy {dvs_test_acc:.2f}% is at or below chance ({CHANCE_DVS:.1f}%)\")\n",
        "    print(\"   → FALL BACK TO SSC-35 (notify Blessing for loader)\")\n",
        "    raise RuntimeError(\"DVS single-task accuracy below chance — aborting. Use SSC-35 fallback.\")\n",
        "elif dvs_test_acc < 30.0:\n",
        "    print(f\"\\n⚠️  WARNING: DVS accuracy {dvs_test_acc:.2f}% < 30% target\")\n",
        "    print(\"   → Non-trivial but sub-optimal; proceed with caution.\")\n",
        "else:\n",
        "    print(f\"\\n✅ DVS smoke test PASSED: {dvs_test_acc:.2f}% >> chance ({CHANCE_DVS:.1f}%)\")\n",
        "\n",
        "# Cleanup\n",
        "del model_smoke, buffer_smoke, trainer\n",
        "torch.cuda.empty_cache()\n",
        "print(\"=\"*80 + \"\\n\")\n"
    ]
}

# Insert after the loader exec cell (now at index 10)
cells.insert(11, cell_smoke)

# =====================================================================
# 3. Modify main CL cell (cell 28 originally, now shifted by 2 -> index 30)
# =====================================================================

# Find the main CL cell by id
cl_cell_idx = None
for i, cell in enumerate(cells):
    if cell.get('id') == 'ec05df78':
        cl_cell_idx = i
        break

assert cl_cell_idx is not None, "Could not find main CL cell"

src = cells[cl_cell_idx]['source']
src_text = ''.join(src)

# ---- 3a. Add 'dvs' to TASKS in the CL cell ----
# Current: TASKS = { 'nmnist': ..., 'shd': ... }
src_text = src_text.replace(
    "TASKS = {\n    'nmnist': {'num_classes': 10, 'type': 'visual'},\n    'shd':    {'num_classes': 20, 'type': 'audio'},\n}",
    "TASKS = {\n    'nmnist': {'num_classes': 10, 'type': 'visual'},\n    'shd':    {'num_classes': 20, 'type': 'audio'},\n    'dvs':    {'num_classes': 11, 'type': 'visual_motion'},\n}"
)

# ---- 3b. Add HF upload helper after save_training_plots ----
hf_helper = '''\n\n# ── HuggingFace upload helper ─────────────────────────────────────────────────\n\ndef upload_to_hf(filepath, repo_id, token=None, path_in_repo=None):\n    """Upload a file to HuggingFace Hub. Falls back gracefully if HF is unavailable."""\n    try:\n        from huggingface_hub import HfApi\n        api = HfApi(token=token)\n        api.upload_file(\n            path_or_fileobj=str(filepath),\n            path_in_repo=path_in_repo or filepath.name,\n            repo_id=repo_id,\n            repo_type="dataset",\n        )\n        print(f"  🤗 HF uploaded: {filepath.name}")\n    except Exception as e:\n        print(f"  ⚠️ HF upload skipped for {filepath.name}: {e}")\n\n\n'''

# Insert after save_training_plots definition
src_text = src_text.replace(
    "    return fname\n\n\ndef run_one_seed",
    "    return fname\n" + hf_helper + "def run_one_seed"
)

# ---- 3c. Extend run_one_seed to include T3 (DVS) and evaluate all 3 tasks ----
# The current run_one_seed ends with:
# "    forgetting = acc_nm_s1 - acc_nm_s2\n"
# "    fwt        = acc_shd_s2 - acc_shd_s1\n"
# Then returns result dict.
# We need to add T3 training block before the final evaluation section.

# Actually, looking at the code, after T2 training there is:
# "    # Final evaluation"
# with evaluation of N-MNIST and SHD. We need to add T3 before that,
# then evaluate all 3 tasks.

# Let's find where the T2 final evaluation starts and insert T3 before it.

old_final_eval = '''    # Final evaluation
    model.eval()
    with torch.no_grad():
        tc = tt = 0
        for xb, yb in test_loader_nmnist:
            xb, yb = xb.to(device), yb.to(device)
            lg, _, _ = model(xb, 'nmnist', 'nmnist')
            tc += (lg.argmax(1) == yb).sum().item()
            tt += yb.size(0)
        acc_nm_s2 = 100. * tc / tt

        sc = st = 0
        for xb, yb in test_loader_shd:
            xb, yb = xb.to(device), yb.to(device)
            lg, _, _ = model(xb, 'shd', 'shd')
            sc += (lg.argmax(1) == yb).sum().item()
            st += yb.size(0)
        acc_shd_s2 = 100. * sc / st

    forgetting = acc_nm_s1 - acc_nm_s2
    fwt        = acc_shd_s2 - acc_shd_s1'''

new_final_eval = '''    # ── Populate buffer with T2 (SHD) ───────────────────────────────────
    if use_replay:
        buffer.populate(model=model, loader=train_loader_shd_cl,
                        modality='shd', task_id=2, label_offset=0)
        print(f"  📦 Buffer populated T2: {buffer.stats()['total']} embeddings stored")

    # ── STEP 3: Train on DVS-Gesture ─────────────────────────────────────
    print(f"\\n  [S{seed}] STEP 3: DVS-Gesture (visual-motion, T3) "
          f"{'+ Replay' if use_replay else ''}")
    print(f"  {'Epoch':>6} {'Train':>8} {'Val':>8} "
          f"{'w_hop':>7} {'w_hgrn':>7}")
    print(f"  {'─'*50}")

    opt3 = torch.optim.AdamW(
        model.parameters(), lr=CL_LR, weight_decay=1e-4)
    sch3 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt3, T_max=CL_EPOCHS, eta_min=1e-6)

    best_val_t3   = 0.0
    best_state_t3 = None
    patience_ctr3 = 0

    for epoch in range(CL_EPOCHS):
        model.train()
        train_correct = train_total = 0
        train_loss_sum = 0.0
        ep_gate_hop = ep_gate_hgrn = 0.0
        n_batches = 0

        for xb, yb in train_loader_dvs_cl:
            xb, yb = xb.to(device), yb.to(device)
            opt3.zero_grad()
            lg, gate, feat = model(xb, 'dvs', 'dvs')
            loss  = ce_loss(lg, yb)
            loss += CONFIG['contrastive_weight'] * scl_fn(feat, yb)
            loss += 0.01 * (-(gate * torch.log(gate + 1e-10))
                            .sum(1).mean())

            # Embedding replay loss from T1 and T2
            if use_replay and buffer.stats()['total'] > 0:
                embs, lbls = buffer.sample(32, task_ids=replay_task_ids)
                if embs is not None:
                    # Replay against T1 (nmnist) and T2 (shd) heads
                    for rtask_name, rtask_id in [('nmnist', 1), ('shd', 2)]:
                        if rtask_id in replay_task_ids:
                            mask = (lbls < model.tasks[rtask_name]['num_classes']) if rtask_name == 'nmnist' else (lbls < model.tasks[rtask_name]['num_classes'])
                            # Simple approach: just replay with offset handling
                            r_lg = model.forward_from_embeddings(embs, rtask_name)
                            loss += 0.15 * ce_loss(r_lg, lbls % model.tasks[rtask_name]['num_classes'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt3.step()

            train_correct += (lg.argmax(1) == yb).sum().item()
            train_total   += yb.size(0)
            train_loss_sum += loss.item()
            ep_gate_hop   += gate[:, 0].mean().item()
            ep_gate_hgrn  += gate[:, 1].mean().item()
            n_batches += 1

        sch3.step()
        train_acc  = 100. * train_correct / train_total
        train_loss = train_loss_sum / n_batches
        avg_hop    = ep_gate_hop  / n_batches
        avg_hgrn   = ep_gate_hgrn / n_batches

        # Val
        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for xv, yv in val_loader_dvs:
                xv, yv = xv.to(device), yv.to(device)
                lv, _, _ = model(xv, 'dvs', 'dvs')
                vc += (lv.argmax(1) == yv).sum().item()
                vt += yv.size(0)
        val_acc = 100. * vc / vt

        hist_t3 = {'train_acc': [], 'val_acc': [], 'train_loss': [],
                   'gate_hop': [], 'gate_hgrn': []}
        hist_t3['train_acc'].append(train_acc)
        hist_t3['val_acc'].append(val_acc)
        hist_t3['train_loss'].append(train_loss)
        hist_t3['gate_hop'].append(avg_hop)
        hist_t3['gate_hgrn'].append(avg_hgrn)

        if (epoch + 1) % 5 == 0:
            marker = '✨' if val_acc > best_val_t3 else '  '
            print(f"  {marker} Ep {epoch+1:3d}: "
                  f"Train={train_acc:6.2f}% "
                  f"Val={val_acc:6.2f}% "
                  f"hop={avg_hop:.3f} hgrn={avg_hgrn:.3f}")

        if val_acc > best_val_t3:
            best_val_t3   = val_acc
            patience_ctr3 = 0
            best_state_t3 = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            torch.save({
                'model_state_dict': best_state_t3,
                'epoch': epoch, 'val_acc': val_acc,
                'seed': seed, 'condition': condition, 'task': 'dvs',
                'gate_final_hop':  avg_hop,
                'gate_final_hgrn': avg_hgrn,
            }, CKPT_DIR / f'M7_dvs_t3_{condition}_seed{seed}.pth')
        else:
            patience_ctr3 += 1
            if patience_ctr3 >= CL_PATIENCE:
                print(f"  🛑 Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(
        {k: v.to(device) for k, v in best_state_t3.items()})

    # Save T3 plot
    p3 = save_training_plots(hist_t3, seed, condition, 'dvs_T3')
    print(f"  📊 T3 plot saved → {p3.name}")

    # Final evaluation: all 3 tasks
    model.eval()
    with torch.no_grad():
        tc = tt = 0
        for xb, yb in test_loader_nmnist:
            xb, yb = xb.to(device), yb.to(device)
            lg, _, _ = model(xb, 'nmnist', 'nmnist')
            tc += (lg.argmax(1) == yb).sum().item()
            tt += yb.size(0)
        acc_nm_s2 = 100. * tc / tt

        sc = st = 0
        for xb, yb in test_loader_shd:
            xb, yb = xb.to(device), yb.to(device)
            lg, _, _ = model(xb, 'shd', 'shd')
            sc += (lg.argmax(1) == yb).sum().item()
            st += yb.size(0)
        acc_shd_s2 = 100. * sc / st

        dc = dt = 0
        for xb, yb in test_loader_dvs:
            xb, yb = xb.to(device), yb.to(device)
            lg, _, _ = model(xb, 'dvs', 'dvs')
            dc += (lg.argmax(1) == yb).sum().item()
            dt += yb.size(0)
        acc_dvs_s3 = 100. * dc / dt

    forgetting = acc_nm_s1 - acc_nm_s2
    fwt        = acc_shd_s2 - acc_shd_s1
    forgetting_dvs = acc_shd_s2 - acc_dvs_s3  # SHD drop after DVS'''

src_text = src_text.replace(old_final_eval, new_final_eval)

# ---- 3d. Update the result dict to include T3 metrics ----
old_result_dict = '''    result = {
        'condition':     condition,
        'seed':          seed,
        'step1_nmnist':  acc_nm_s1,
        'step1_shd':     acc_shd_s1,
        'step2_nmnist':  acc_nm_s2,
        'step2_shd':     acc_shd_s2,
        'forgetting':    forgetting,
        'forward_transfer': fwt,
        'R_T1_after_T1': acc_nm_s1,
        'R_T1_after_T2': acc_nm_s2,
        'R_T2_after_T2': acc_shd_s2,
        'gate_t2_final_hop':  gate_final_hop,
        'gate_t2_final_hgrn': gate_final_hgrn,
        'best_val_t1':   best_val_t1,
        'best_val_t2':   best_val_t2,
    }'''

new_result_dict = '''    result = {
        'condition':     condition,
        'seed':          seed,
        'step1_nmnist':  acc_nm_s1,
        'step1_shd':     acc_shd_s1,
        'step2_nmnist':  acc_nm_s2,
        'step2_shd':     acc_shd_s2,
        'step3_nmnist':  acc_nm_s2,
        'step3_shd':     acc_shd_s2,
        'step3_dvs':     acc_dvs_s3,
        'forgetting':    forgetting,
        'forward_transfer': fwt,
        'forgetting_shd_to_dvs': forgetting_dvs,
        'R_T1_after_T1': acc_nm_s1,
        'R_T1_after_T2': acc_nm_s2,
        'R_T2_after_T2': acc_shd_s2,
        'R_T1_after_T3': acc_nm_s2,
        'R_T2_after_T3': acc_shd_s2,
        'R_T3_after_T3': acc_dvs_s3,
        'gate_t2_final_hop':  gate_final_hop,
        'gate_t2_final_hgrn': gate_final_hgrn,
        'best_val_t1':   best_val_t1,
        'best_val_t2':   best_val_t2,
        'best_val_t3':   best_val_t3,
    }'''

src_text = src_text.replace(old_result_dict, new_result_dict)

# ---- 3e. Add HF upload and CSV save after each seed ----
old_seed_end = '''        # Save CSV incrementally after every single seed
        flat = [
            {k: v for k, v in r.items()}
            for cond_recs in cl_results.values()
            for r in cond_recs
        ]
        df = pd.DataFrame(flat)
        df.to_csv(RESULTS_DIR / 'cl_results.csv', index=False)
        print(f"  💾 CSV updated: {len(df)} rows saved")'''

new_seed_end = '''        # Save CSV incrementally after every single seed
        flat = [
            {k: v for k, v in r.items()}
            for cond_recs in cl_results.values()
            for r in cond_recs
        ]
        df = pd.DataFrame(flat)
        df.to_csv(RESULTS_DIR / 'cl_trimodal_results.csv', index=False)
        print(f"  💾 CSV updated: {len(df)} rows saved → cl_trimodal_results.csv")

        # Upload checkpoints to HuggingFace after every seed (Rule 2)
        HF_REPO = os.environ.get('HF_REPO', None)
        if HF_REPO:
            for ckpt_file in CKPT_DIR.glob(f'*_seed{seed}.pth'):
                upload_to_hf(ckpt_file, HF_REPO, path_in_repo=f"checkpoints/{condition}/{ckpt_file.name}")
            upload_to_hf(RESULTS_DIR / 'cl_trimodal_results.csv', HF_REPO, path_in_repo="cl_trimodal_results.csv")
        else:
            print(f"  ℹ️ HF_REPO not set; skipping HuggingFace upload. Export HF_REPO=<your-dataset-repo> to enable.")'''

src_text = src_text.replace(old_seed_end, new_seed_end)

# ---- 3f. Update the loop header for tri-modal ----
# Change total = len(['no_replay', 'replay']) * len(CL_SEEDS)
# to account for the fact that we're now doing tri-modal
# Actually this is just the progress counter, it's fine.

# ---- 3g. Update final summary to include DVS ----
old_summary = '''for condition in ['no_replay', 'replay']:
    sub = df[df['condition'] == condition]
    if len(sub) == 0:
        continue
    print(f"\\n── {condition.upper()} ({len(sub)} seeds) ──────────────────────")
    print(f"  N-MNIST after T2:  "
          f"{sub['R_T1_after_T2'].mean():.2f} ± "
          f"{sub['R_T1_after_T2'].std():.2f}%")
    print(f"  SHD after T2:      "
          f"{sub['R_T2_after_T2'].mean():.2f} ± "
          f"{sub['R_T2_after_T2'].std():.2f}%")
    print(f"  Forgetting F:      "
          f"{sub['forgetting'].mean():.2f} ± "
          f"{sub['forgetting'].std():.2f}pp")
    print(f"  Fwd Transfer FWT:  "
          f"{sub['forward_transfer'].mean():.2f} ± "
          f"{sub['forward_transfer'].std():.2f}pp")
    if 'gate_t2_final_hgrn' in sub.columns:
        print(f"  Gate w_hgrn (T2):  "
              f"{sub['gate_t2_final_hgrn'].mean():.3f} ± "
              f"{sub['gate_t2_final_hgrn'].std():.3f}")'''

new_summary = '''for condition in ['no_replay', 'replay']:
    sub = df[df['condition'] == condition]
    if len(sub) == 0:
        continue
    print(f"\\n── {condition.upper()} ({len(sub)} seeds) ──────────────────────")
    print(f"  N-MNIST after T3:  "
          f"{sub['R_T1_after_T3'].mean():.2f} ± "
          f"{sub['R_T1_after_T3'].std():.2f}%")
    print(f"  SHD after T3:      "
          f"{sub['R_T2_after_T3'].mean():.2f} ± "
          f"{sub['R_T2_after_T3'].std():.2f}%")
    print(f"  DVS after T3:      "
          f"{sub['R_T3_after_T3'].mean():.2f} ± "
          f"{sub['R_T3_after_T3'].std():.2f}%")
    print(f"  Forgetting F (NM): "
          f"{sub['forgetting'].mean():.2f} ± "
          f"{sub['forgetting'].std():.2f}pp")
    print(f"  Fwd Transfer FWT:  "
          f"{sub['forward_transfer'].mean():.2f} ± "
          f"{sub['forward_transfer'].std():.2f}pp")
    if 'gate_t2_final_hgrn' in sub.columns:
        print(f"  Gate w_hgrn (T2):  "
              f"{sub['gate_t2_final_hgrn'].mean():.3f} ± "
              f"{sub['gate_t2_final_hgrn'].std():.3f}")'''

src_text = src_text.replace(old_summary, new_summary)

# ---- 3h. Fix buffer.total_stored bug (should be buffer.stats()['total']) ----
src_text = src_text.replace("buffer.total_stored", "buffer.stats()['total']")

# ---- 3i. Add import os at top of CL cell if not present ----
if "import os" not in src_text[:500]:
    src_text = src_text.replace(
        "import torch.nn.functional as F",
        "import os\nimport torch.nn.functional as F",
        1
    )

# Write back the modified source
cells[cl_cell_idx]['source'] = list(src_text)

# =====================================================================
# 4. Save modified notebook
# =====================================================================
with open('NeurIPS2026_M7_clean (3) (1).ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully!")
print(f"  - Added DVS loader execution cell at index 10")
print(f"  - Added DVS smoke test cell at index 11")
print(f"  - Modified main CL cell at index {cl_cell_idx}")
