# Experiment Log

## 2026-04-29

### Changes Made
- [x] Added DVS-Gesture loader execution cell (Cell 8.5b)
- [x] Added DVS single-task smoke test (Cell 8.5c)
- [x] Extended main CL cell for tri-modal T3
- [x] Added HF upload after every seed
- [x] Fixed `buffer.total_stored` → `buffer.stats()['total']`
- [x] Fixed `buffer.sample` unpacking (2 values, not 3)
- [x] Fixed `hist_t3` initialization (before loop, not inside)
- [x] Fixed forgetting metrics (`forgetting_dvs` was comparing SHD vs DVS accuracy)

### Next Runs
- [ ] Run smoke test with seed 42
- [ ] Confirm DVS accuracy > 30%
- [ ] Run full tri-modal CL (no_replay + replay, 3 seeds)
- [ ] Upload results to HuggingFace

## Results Template

```markdown
### Seed {N} | {Condition}
- N-MNIST after T1: __%
- N-MNIST after T2: __%
- SHD after T2: __%
- N-MNIST after T3: __%
- SHD after T3: __%
- DVS after T3: __%
- Forgetting T1→T2: __pp
- FWT T2: __pp
- Forgetting NM→T3: __pp
- Forgetting SHD→T3: __pp
- Gate w_hgrn (T2): __
```
