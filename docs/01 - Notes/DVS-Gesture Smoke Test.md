# DVS-Gesture Smoke Test

## Purpose
Validate that DVS-Gesture single-task training works before running the expensive tri-modal CL sequence.

## Cell Location
Notebook: `NeurIPS2026_M7_clean (3) (1).ipynb` — **Cell 11** (`dvs-smoke-test`)

## What It Does

1. Seeds everything with `42`
2. Builds temporary `M7_ContinualAdaptiveModel` with only the `'dvs'` task
3. Creates `ContinualLearner` with empty replay buffer
4. Runs `trainer.train_task(...)` for 30 epochs
5. Asserts tensor shapes:
   - Input: `[B, 25, 2, 34, 34]`
   - Logits: `[B, 11]`
   - Gate: `[B, 2]`
   - Features: `[B, 512]`
6. Checks accuracy against thresholds

## Accuracy Guards

```python
CHANCE_DVS = 100.0 / 11.0  # ~9.09%

if dvs_test_acc <= CHANCE_DVS + 1.0:   # <= ~10.1%
    raise RuntimeError("DVS single-task accuracy below chance — aborting.")
elif dvs_test_acc < 30.0:              # < 30%
    print("WARNING: sub-optimal; proceed with caution.")
else:
    print("PASS")
```

## Fallback

If DVS download fails or accuracy is below chance:
> **Use SSC-35** (notify Blessing for loader)

## Cleanup

The cell deletes `model_smoke`, `buffer_smoke`, `trainer` and calls `torch.cuda.empty_cache()`.

> ⚠️ Run this cell **before** the main CL cell (Cell 30). The main CL cell instantiates its own fresh model/buffer/learner.
