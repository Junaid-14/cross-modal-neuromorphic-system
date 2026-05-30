# Team Rules & Runbook

## Rule 1: Run Cells in Order

The notebook has dependencies between cells. Do not skip or reorder.

```
Cells 1→9    : Setup + N-MNIST + SHD loaders
Cell 10      : DVS loader + val split
Cell 11      : DVS smoke test (must pass before main CL)
Cells 12→29  : Model definitions + training pipeline
Cell 30      : Main tri-modal CL (~4–5 hrs)
Cells 31→54  : Analysis, figures, baselines
```

## Rule 2: Save to HuggingFace After Every Seed

Do not trust RunPod local storage. Upload after **every single seed**.

```bash
export HF_REPO="your-username/your-dataset-repo"
```

The notebook will auto-upload:
- Checkpoints (`*.pth`)
- `cl_trimodal_results.csv`

If `HF_REPO` is not set, uploads are skipped with a warning.

## Rule 3: Flag Immediately on Failure

### DVS-Gesture Download Failure
- `tonic.datasets.DVSGesture` download can fail
- The notebook wraps the loader in `try/except`
- **Fallback**: SSC-35 (Blessing has the loader)
- Action: Notify Blessing, swap dataset, re-run

### Below-Chance Accuracy
- DVS chance on 11 classes = **9.09%**
- Smoke test raises `RuntimeError` if `acc ≤ 10.1%`
- **Fallback**: SSC-35
- Action: Check data preprocessing, swap dataset

### Sub-Optimal Accuracy
- Target: **> 30%** on DVS single-task
- If `10.1% < acc < 30%`: warning printed, proceed with caution

## Environment Checklist

- [ ] GPU available (`torch.cuda.is_available()`)
- [ ] `HF_REPO` env var set (for uploads)
- [ ] Datasets directory writable (`ICONS_M7/datasets/`)
- [ ] ~30 GB disk space (for datasets + checkpoints)
- [ ] ~25 GB GPU memory (RTX 4090 or better)

## Emergency Contacts

| Issue | Contact |
|-------|---------|
| DVS loader / SSC-35 | Blessing |
| HF upload failures | DevOps |
| Notebook crashes | Team lead |
| Tonic download issues | Check IBM/tonic GitHub |
