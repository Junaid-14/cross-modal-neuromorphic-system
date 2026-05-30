# Datasets

## N-MNIST (T1)
- **Modality**: Visual
- **Sensor**: 34×34, 2 polarity channels
- **Classes**: 10
- **Train**: 60,000 | **Test**: 10,000
- **Preprocessing**: Denoise(filter_time=10000) → ToFrame(time_window=1000)
- **Collate**: Pad/truncate to T=25
- **Val split**: 10% from train (6,000)

## SHD (T2)
- **Modality**: Audio
- **Channels**: 700
- **Classes**: 20
- **Train**: 8,156 | **Test**: 2,264
- **Preprocessing**: Events → dense (100 time bins, 700 channels)
- **Val split**: 10% from train (815)

## DVS-Gesture (T3)
- **Modality**: Visual Motion
- **Sensor**: 128×128, 2 polarity channels
- **Classes**: 11
- **Train**: ~1,000 | **Test**: ~300
- **Preprocessing**: Denoise(filter_time=10000) → ToFrame(time_window=5000)
- **Spatial**: Adaptive pool 128×128 → 34×34
- **Temporal**: Pad/truncate to T=25
- **Val split**: 10% from train
- **Fallback**: SSC-35 (Blessing) if download fails

## Download Risks
| Dataset | Source | Risk |
|---------|--------|------|
| N-MNIST | S3 (EU) | Usually stable |
| SHD | zenkelab.org | Usually stable |
| DVS-Gesture | IBM/tonic | ⚠️ Can fail; fallback to SSC-35 |
