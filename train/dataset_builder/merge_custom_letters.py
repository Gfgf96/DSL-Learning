"""
Replace H, P, T, W in Dataset
=============================
1. Load custom letters
2. Augment x10
3. Remove old H, P, T, W from main dataset
4. Add new ones
5. Normalize everything
6. Save final dataset
"""

import numpy as np
from pathlib import Path

# Paths (adjust to your setup)
CUSTOM_DIR = Path("./data_collect/ngt_custom")
MAIN_DATASET = Path("./data_collect/ngt_frankenstein_x10.npz")
OUTPUT_PATH = Path("./data_collect/ngt_frankenstein_final.npz")

LETTERS_TO_REPLACE = ['H', 'P', 'T', 'W']
AUGMENT_MULTIPLIER = 10


# ============ AUGMENTATION ============

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
    """Translate landmarks."""
    points = coords.reshape(-1, 3)
    points[:, 0] += np.random.uniform(-max_shift, max_shift)
    points[:, 1] += np.random.uniform(-max_shift, max_shift)
    return points.flatten()


def augment_sample(coords):
    """Apply random augmentations to single sample."""
    aug = coords.copy()
    aug = add_noise(aug, std=np.random.uniform(0.005, 0.015))
    if np.random.random() < 0.7:
        aug = scale(aug)
    if np.random.random() < 0.7:
        aug = rotate_2d(aug)
    if np.random.random() < 0.5:
        aug = translate(aug)
    return aug


# ============ NORMALIZATION ============

def normalize_landmarks(coords):
    """
    Normalize landmarks:
    1. Center on wrist (point 0)
    2. Scale by hand size (distance to middle finger base)
    """
    points = coords.reshape(21, 3)
    
    # Center on wrist
    wrist = points[0].copy()
    points = points - wrist
    
    # Scale by hand size (point 9 = middle finger base)
    scale_factor = np.linalg.norm(points[9])
    if scale_factor > 0.001:
        points = points / scale_factor
    
    return points.flatten()


# ============ MAIN PIPELINE ============

def main():
    print("=" * 60)
    print("🔧 REPLACING H, P, T, W IN DATASET")
    print("=" * 60)
    
    # 1. Load custom letters
    print("\n📂 1. Loading custom letters...")
    custom_data = {}
    for letter in LETTERS_TO_REPLACE:
        path = CUSTOM_DIR / f"{letter}_landmarks.npy"
        if path.exists():
            data = np.load(path)
            custom_data[letter] = data
            print(f"   ✅ {letter}: {len(data)} samples")
        else:
            print(f"   ❌ {letter}: file not found!")
    
    if not custom_data:
        print("\n❌ No custom data found! Exiting.")
        return
    
    # 2. Augment custom letters
    print(f"\n🔄 2. Augmenting x{AUGMENT_MULTIPLIER}...")
    custom_augmented_X = []
    custom_augmented_y = []
    
    for letter, data in custom_data.items():
        augmented = list(data)  # start with originals
        for _ in range(AUGMENT_MULTIPLIER - 1):
            for sample in data:
                augmented.append(augment_sample(sample))
        
        custom_augmented_X.extend(augmented)
        custom_augmented_y.extend([letter] * len(augmented))
        print(f"   {letter}: {len(data)} → {len(augmented)}")
    
    custom_X = np.array(custom_augmented_X, dtype=np.float32)
    custom_y = np.array(custom_augmented_y)
    
    # 3. Load main dataset and remove old letters
    print(f"\n📂 3. Loading main dataset...")
    main_data = np.load(MAIN_DATASET, allow_pickle=True)
    X_main = main_data['X']
    y_main = main_data['y']
    print(f"   Total: {len(X_main)} samples")
    
    # Count how many we'll remove
    for letter in LETTERS_TO_REPLACE:
        count = (y_main == letter).sum()
        print(f"   Removing {letter}: {count} samples")
    
    # Filter - keep everything except H, P, T, W
    mask = ~np.isin(y_main, LETTERS_TO_REPLACE)
    X_filtered = X_main[mask]
    y_filtered = y_main[mask]
    print(f"   After removal: {len(X_filtered)} samples")
    
    # 4. Merge
    print("\n🔗 4. Merging...")
    X_combined = np.vstack([X_filtered, custom_X])
    y_combined = np.concatenate([y_filtered, custom_y])
    print(f"   Total: {len(X_combined)} samples")
    
    # 5. Normalize ALL
    print("\n📐 5. Normalizing...")
    X_normalized = np.array([normalize_landmarks(x) for x in X_combined], dtype=np.float32)
    
    # 6. Shuffle
    print("\n🔀 6. Shuffling...")
    indices = np.random.permutation(len(X_normalized))
    X_final = X_normalized[indices]
    y_final = y_combined[indices]
    
    # 7. Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 7. Saving: {OUTPUT_PATH}")
    np.savez(OUTPUT_PATH, X=X_final, y=y_final)
    
    # Also save CSV
    csv_path = str(OUTPUT_PATH).replace('.npz', '.csv')
    import pandas as pd
    columns = ['label'] + [f'coord_{i}' for i in range(63)]
    df = pd.DataFrame(np.column_stack([y_final, X_final]), columns=columns)
    df.to_csv(csv_path, index=False)
    print(f"💾 CSV: {csv_path}")
    
    # Report
    print("\n" + "=" * 60)
    print("📊 FINAL REPORT")
    print("=" * 60)
    print(f"   Total samples: {len(X_final)}")
    print(f"   Classes: {len(set(y_final))}")
    print("\n   Per letter:")
    from collections import Counter
    counts = Counter(y_final)
    for letter in sorted(counts.keys()):
        marker = "🆕" if letter in LETTERS_TO_REPLACE else "  "
        print(f"   {marker} {letter}: {counts[letter]}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()