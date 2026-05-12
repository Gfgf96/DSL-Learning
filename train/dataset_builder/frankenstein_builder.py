"""
NGT Frankenstein Dataset Builder
================================
Creates composite landmarks dataset for NGT (Dutch Sign Language).
- ASL: extracts landmarks from images via MediaPipe
- DGS: uses pre-extracted landmarks from CSV
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =============================================================================
# NGT COMPATIBILITY MATRIX (from research report)
# =============================================================================

LETTER_SOURCE = {
    'A': 'asl', 'B': 'asl', 'C': 'asl', 'D': 'asl',
    'E': 'mixed',  # Mix ASL + DGS
    'F': 'asl',
    'G': 'dgs',    # ⚠️ CRITICAL! ASL is horizontal
    'H': 'dgs',    # ⚠️ CRITICAL! Orientation differs
    'I': 'asl',
    # J - excluded (dynamic gesture)
    'K': 'asl',
    'L': 'asl',
    'M': 'dgs',    # ⚠️ CRITICAL! Different configuration
    'N': 'dgs',    # ⚠️ CRITICAL! Different configuration
    'O': 'asl',
    'P': 'dgs',    # Unique form (ideally LSF, but using DGS)
    'Q': 'dgs',    # Orientation issue
    'R': 'asl', 'S': 'asl',
    'T': 'dgs',    # ⚠️ CRITICAL! ASL T is obscene in Netherlands!
    'U': 'asl', 'V': 'asl', 'W': 'asl', 'X': 'asl', 'Y': 'asl',
    # Z - excluded (dynamic gesture)
}

CRITICAL_LETTERS = {'G', 'H', 'M', 'N', 'T', 'P', 'Q'}

# MediaPipe model path
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "mediapipe" / "hand_landmarker.task"


class FrankensteinBuilder:
    def __init__(
        self,
        asl_images_path: str,
        dgs_csv_path: str,
        output_path: str = "./ngt_frankenstein",
        samples_per_class: int = 500
    ):
        self.asl_path = Path(asl_images_path)
        self.dgs_csv = Path(dgs_csv_path)
        self.output_path = Path(output_path)
        self.samples_per_class = samples_per_class
        
        # MediaPipe Tasks API
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Load DGS
        print("📂 Loading DGS landmarks...")
        self.dgs_df = pd.read_csv(self.dgs_csv)
        self.dgs_df['label'] = self.dgs_df['label'].str.upper()
        print(f"   DGS: {len(self.dgs_df)} samples")
    
    def extract_landmarks_from_image(self, img_path: Path) -> np.ndarray | None:
        """Extract 21 landmarks (63 coordinates) from image."""
        mp_image = mp.Image.create_from_file(str(img_path))
        results = self.detector.detect(mp_image)
        
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand = results.hand_landmarks[0]
            coords = []
            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords)
        return None
    
    def extract_asl_landmarks(self, letter: str, max_samples: int) -> np.ndarray:
        """Extract landmarks for letter from ASL images."""
        letter_path = self.asl_path / letter.upper()
        if not letter_path.exists():
            print(f"   ⚠️ Folder {letter} not found in ASL")
            return np.array([])
        
        images = list(letter_path.glob("*.jpg")) + list(letter_path.glob("*.png"))
        images = images[:max_samples * 2]  # Take extra (not all will be detected)
        
        landmarks = []
        for img_path in tqdm(images, desc=f"   ASL {letter}", leave=False):
            lm = self.extract_landmarks_from_image(img_path)
            if lm is not None:
                landmarks.append(lm)
            if len(landmarks) >= max_samples:
                break
        
        return np.array(landmarks) if landmarks else np.array([])
    
    def get_dgs_landmarks(self, letter: str, max_samples: int) -> np.ndarray:
        """Get landmarks for letter from DGS CSV."""
        letter_df = self.dgs_df[self.dgs_df['label'] == letter.upper()]
        if len(letter_df) == 0:
            print(f"   ⚠️ Letter {letter} not found in DGS")
            return np.array([])
        
        # Take only coordinates (without label)
        coords = letter_df.iloc[:max_samples, 1:].values
        return coords
    
    def build(self):
        """Build composite dataset."""
        print("=" * 60)
        print("🧟 NGT FRANKENSTEIN DATASET BUILDER")
        print("=" * 60)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        all_data = []
        stats = {}
        
        for letter, source in LETTER_SOURCE.items():
            critical = "🔴" if letter in CRITICAL_LETTERS else "  "
            print(f"\n{critical} Letter {letter} ← {source.upper()}")
            
            if source == 'asl':
                landmarks = self.extract_asl_landmarks(letter, self.samples_per_class)
                
            elif source == 'dgs':
                landmarks = self.get_dgs_landmarks(letter, self.samples_per_class)
                if len(landmarks) == 0:
                    # Fallback to ASL with warning
                    print(f"   ⚠️ DGS empty, fallback to ASL (MAY CAUSE ERRORS!)")
                    landmarks = self.extract_asl_landmarks(letter, self.samples_per_class)
                    
            elif source == 'mixed':
                # 70% ASL + 30% DGS
                n_asl = int(self.samples_per_class * 0.7)
                n_dgs = self.samples_per_class - n_asl
                asl_lm = self.extract_asl_landmarks(letter, n_asl)
                dgs_lm = self.get_dgs_landmarks(letter, n_dgs)
                landmarks = np.vstack([asl_lm, dgs_lm]) if len(dgs_lm) > 0 else asl_lm
            
            # Add labels
            for lm in landmarks:
                all_data.append([letter] + list(lm))
            
            stats[letter] = {'source': source, 'count': len(landmarks)}
            print(f"   ✅ {len(landmarks)} samples")
        
        # Create DataFrame
        columns = ['label'] + [f'coord_{i}' for i in range(63)]
        df = pd.DataFrame(all_data, columns=columns)
        
        # Save CSV
        csv_path = self.output_path / "ngt_frankenstein_landmarks.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Saved: {csv_path}")
        
        # Also save as numpy
        np.savez(
            self.output_path / "ngt_frankenstein.npz",
            X=df.iloc[:, 1:].values.astype(np.float32),
            y=df['label'].values
        )
        
        self._print_report(stats, df)
        return df
    
    def _print_report(self, stats: dict, df: pd.DataFrame):
        """Print final report."""
        print("\n" + "=" * 60)
        print("📊 REPORT")
        print("=" * 60)
        
        print(f"\n📁 Path: {self.output_path}")
        print(f"📈 Total samples: {len(df):,}")
        print(f"🔤 Classes: {df['label'].nunique()}")
        
        print("\n📋 Per letter:")
        print("-" * 40)
        for letter in sorted(stats.keys()):
            data = stats[letter]
            critical = "🔴" if letter in CRITICAL_LETTERS else "  "
            print(f"  {critical} {letter}: {data['count']:4d} | {data['source']}")
        
        # Warnings
        print("\n⚠️ REMINDER:")
        print("   Letters G, H, M, N, P, Q, T taken from DGS")
        print("   (ASL versions are NOT compatible with NGT!)")
        print("\n   J and Z excluded (dynamic gestures)")


# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":
    # hand_landmarker.task is downloaded automatically via downloader script
    # or manually via:
    # curl -L -o ../../models/mediapipe/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    
    builder = FrankensteinBuilder(
        asl_images_path="./datasets/asl/asl_alphabet_train/asl_alphabet_train",
        dgs_csv_path="./datasets/dgs/german_sign_language.csv",
        output_path="./datasets/ngt_frankenstein",
        samples_per_class=500  # Can increase to 1000+
    )
    
    df = builder.build()
    print("\n✅ Done!")