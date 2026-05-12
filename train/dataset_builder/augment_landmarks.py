"""
Augment Landmarks Dataset x10
=============================
Takes ngt_frankenstein.npz and creates ngt_frankenstein_x10.npz
"""

import numpy as np
from pathlib import Path

INPUT_PATH = Path("./datasets/ngt_frankenstein/ngt_frankenstein.npz")
OUTPUT_PATH = Path("./datasets/ngt_frankenstein/ngt_frankenstein_x10.npz")

AUGMENT_MULTIPLIER = 10


# ============ AUGMENTATION FUNCTIONS ============

def add_noise(coords, std=0.01):
    """Add random Gaussian noise."""
    return coords + np.random.normal(0, std, coords.shape)


def scale(coords, factor_range=(0.85, 1.15)):
    """Scale landmarks around center."""
    factor = np.random.uniform(*factor_range)
    points = coords.reshape(-1, 3)
    center = points.mean(axis=0)
    return ((points - center) * factor + center).flatten()


def rotate_2d(coords, max_angle=20):
    """Rotate landmarks in XY plane."""
    angle = np.radians(np.random.uniform(-max_angle, max_angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    points = coords.reshape(-1, 3)
    x, y = points[:, 0], points[:, 1]
    cx, cy = x.mean(), y.mean()
    x_new = (x - cx) * cos_a - (y - cy) * sin_a + cx
    y_new = (x - cx) * sin_a + (y - cy) * cos_a + cy
    points[:, 0], points[:, 1] = x_new, y_new
    return points.flatten()


def translate(coords, max_shift=0.08):
    """Translate landmarks randomly."""
    points = coords.reshape(-1, 3)
    points[:, 0] += np.random.uniform(-max_shift, max_shift)
    points[:, 1] += np.random.uniform(-max_shift, max_shift)
    return points.flatten()


def mirror(coords):
    """Horizontal flip (left/right hand)."""
    points = coords.reshape(-1, 3)
    points[:, 0] = 1.0 - points[:, 0]
    return points.flatten()


def augment_sample(coords):
    """Apply random augmentations to single sample."""
    aug = coords.copy()
    
    # Always add some noise
    aug = add_noise(aug, std=np.random.uniform(0.005, 0.015))
    
    # Random augmentations
    if np.random.random() < 0.7:
        aug = scale(aug)
    if np.random.random() < 0.7:
        aug = rotate_2d(aug)
    if np.random.random() < 0.5:
        aug = translate(aug)
    if np.random.random() < 0.3:
        aug = mirror(aug)
    
    return aug


# ============ MAIN ============

def main():
    print("=" * 60)
    print(f"🔄 AUGMENTING LANDMARKS x{AUGMENT_MULTIPLIER}")
    print("=" * 60)
    
    # Load
    print(f"\n📂 Loading: {INPUT_PATH}")
    data = np.load(INPUT_PATH, allow_pickle=True)
    X_orig = data['X']
    y_orig = data['y']
    print(f"   Original: {len(X_orig)} samples")
    
    # Augment
    print(f"\n🔄 Augmenting...")
    X_aug = list(X_orig)  # start with originals
    y_aug = list(y_orig)
    
    for i in range(AUGMENT_MULTIPLIER - 1):
        print(f"   Pass {i+2}/{AUGMENT_MULTIPLIER}...")
        for x, label in zip(X_orig, y_orig):
            X_aug.append(augment_sample(x))
            y_aug.append(label)
    
    X_final = np.array(X_aug, dtype=np.float32)
    y_final = np.array(y_aug)
    
    # Shuffle
    print("\n🔀 Shuffling...")
    indices = np.random.permutation(len(X_final))
    X_final = X_final[indices]
    y_final = y_final[indices]
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Saving: {OUTPUT_PATH}")
    np.savez(OUTPUT_PATH, X=X_final, y=y_final)
    
    # Also CSV
    csv_path = str(OUTPUT_PATH).replace('.npz', '.csv')
    import pandas as pd
    columns = ['label'] + [f'coord_{i}' for i in range(63)]
    df = pd.DataFrame(np.column_stack([y_final, X_final]), columns=columns)
    df.to_csv(csv_path, index=False)
    print(f"💾 CSV: {csv_path}")
    
    # Report
    print("\n" + "=" * 60)
    print("📊 RESULT")
    print("=" * 60)
    print(f"   Original: {len(X_orig):,} samples")
    print(f"   Augmented: {len(X_final):,} samples")
    print(f"   Classes: {len(set(y_final))}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()