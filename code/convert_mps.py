#!/usr/bin/env python3
"""Convert NeurIPS M7 notebook script to MPS-compatible version."""
import re

with open('m7_raw.py', 'r') as f:
    code = f.read()

# =============================================================================
# 1. Device selection — replace the cuda-only logic with MPS-aware logic
# =============================================================================

old_device_block = '''device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')'''
new_device_block = '''if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')'''
code = code.replace(old_device_block, new_device_block)

# Also handle the commented version in Cell 30
old_comment = '''# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'''
new_comment = '''# device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))'''
code = code.replace(old_comment, new_comment)

# =============================================================================
# 2. torch.cuda.manual_seed_all → torch.manual_seed (works for MPS/CPU/CUDA)
# =============================================================================

code = code.replace('torch.cuda.manual_seed_all(seed)', 'torch.manual_seed(seed)')

# =============================================================================
# 3. torch.cuda.empty_cache → safe no-op on MPS
# =============================================================================

# Replace standalone calls with a safe wrapper
code = code.replace('torch.cuda.empty_cache()', 'safe_empty_cache()')

# Add helper function after the first torch import
helper_code = '''
# ── MPS/CPU compatibility helpers ────────────────────────────────────────────
def safe_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # MPS has no empty_cache API; OS manages unified memory

'''

# Insert helper after the first `import torch` line
if 'import torch' in code:
    # Find first import torch line
    lines = code.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('import torch'):
            insert_idx = i + 1
            break
    lines.insert(insert_idx, helper_code)
    code = '\n'.join(lines)

# =============================================================================
# 4. Replace hardcoded `device='cuda'` defaults in function signatures
# =============================================================================

# Pattern: device='cuda' in function definitions → device=None, then auto-detect inside
# We'll use regex to find these and replace carefully

def replace_cuda_defaults(text):
    # Match device='cuda' in function defs
    pattern = r"(def \w+\([^)]*)device='cuda'([^)]*\):)"
    
    def replacer(match):
        prefix = match.group(1)
        suffix = match.group(2)
        # Replace device='cuda' with device=None
        prefix = prefix.replace("device='cuda'", "device=None")
        return prefix + suffix
    
    text = re.sub(pattern, replacer, text)
    return text

code = replace_cuda_defaults(code)

# For classes that had device='cuda' and now have device=None,
# we need to add auto-detection inside their __init__.
# We'll do a targeted insertion for EmbeddingReplayBuffer and ContinualLearner.

# EmbeddingReplayBuffer
old_er_init = '''    def __init__(self, feature_dim: int = 512,
                 samples_per_class: int = 50,
                 device=None):'''
new_er_init = '''    def __init__(self, feature_dim: int = 512,
                 samples_per_class: int = 50,
                 device=None):
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))'''
code = code.replace(old_er_init, new_er_init)

# ContinualLearner
old_cl_init = '''    def __init__(self, model, buffer, device=None,
                 lr=1e-3, weight_decay=1e-4, gradient_clip=1.0):'''
new_cl_init = '''    def __init__(self, model, buffer, device=None,
                 lr=1e-3, weight_decay=1e-4, gradient_clip=1.0):
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))'''
code = code.replace(old_cl_init, new_cl_init)

# Any remaining device=None that wasn't handled — add a guard at top of file
# Actually, let's also handle the `def __init__(..., device=None, ...)` cases
# where we need to add the auto-detect logic inside the method body.

# Let's find all classes with device=None and add auto-detect if missing
lines = code.split('\n')
new_lines = []
i = 0
while i < len(lines):
    new_lines.append(lines[i])
    # If we see a def __init__ with device=None, check if next non-empty line has auto-detect
    if 'def __init__' in lines[i] and 'device=None' in lines[i]:
        # Look ahead to see if auto-detect is already there
        j = i + 1
        found_auto = False
        while j < len(lines) and lines[j].strip() != '' and not lines[j].strip().startswith('def '):
            if 'torch.backends.mps.is_available()' in lines[j]:
                found_auto = True
                break
            j += 1
        if not found_auto:
            # Need to insert auto-detect after the def line
            # Find the indent level
            indent = len(lines[i]) - len(lines[i].lstrip()) + 4
            auto_line = ' ' * indent + "if device is None:\n" + ' ' * (indent + 4) + "device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))"
            new_lines.append(auto_line)
    i += 1
code = '\n'.join(new_lines)

# =============================================================================
# 5. Handle CUDA-specific print statements gracefully
# =============================================================================

# Replace the CUDA info block in Cell 4/5 with backend-agnostic info
old_cuda_info = '''    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")'''
new_cuda_info = '''    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.backends.mps.is_available():
        print("GPU: Apple Silicon MPS")
    elif torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")'''
code = code.replace(old_cuda_info, new_cuda_info)

# Also handle the second CUDA info block
old_cuda_info2 = '''    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")'''
new_cuda_info2 = '''    if torch.backends.mps.is_available():
        print("   GPU: Apple Silicon MPS")
        print("   Memory: Shared unified memory (OS managed)")
    elif torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")'''
code = code.replace(old_cuda_info2, new_cuda_info2)

# =============================================================================
# 6. Write output
# =============================================================================

with open('m7_mps.py', 'w') as f:
    f.write(code)

print("Created m7_mps.py")
print("\nKey changes made:")
print("  1. Device selection now prefers MPS > CUDA > CPU")
print("  2. torch.cuda.manual_seed_all → torch.manual_seed")
print("  3. torch.cuda.empty_cache → safe_empty_cache() (no-op on MPS)")
print("  4. Hardcoded device='cuda' defaults → device=None with auto-detect")
print("  5. CUDA-specific print blocks → backend-agnostic info")
