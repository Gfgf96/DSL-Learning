"""
NGT Letter Recorder
====================
Records landmarks for letters H, P, T, W

Controls:
  SPACE - save current frame
  1,2,3,4 - switch letter (H,P,T,W)
  Q - quit and save
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import time

# Letters to record
LETTERS = ['H', 'P', 'T', 'W']
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "mediapipe" / "hand_landmarker.task"
OUTPUT_DIR = Path(__file__).parent / "ngt_custom"

# MediaPipe drawing (for visualization)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]


class LandmarkRecorder:
    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # MediaPipe detector
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Data
        self.data = {letter: [] for letter in LETTERS}
        self.current_letter_idx = 0
        self.last_save_time = 0
        
        # Load existing data if available
        self.load_existing()
    
    def load_existing(self):
        """Load previously recorded data."""
        for letter in LETTERS:
            path = OUTPUT_DIR / f"{letter}_landmarks.npy"
            if path.exists():
                existing = np.load(path)
                self.data[letter] = list(existing)
                print(f"📂 Loaded {letter}: {len(self.data[letter])} samples")
    
    def save_all(self):
        """Save all data."""
        for letter in LETTERS:
            if self.data[letter]:
                arr = np.array(self.data[letter])
                np.save(OUTPUT_DIR / f"{letter}_landmarks.npy", arr)
                print(f"💾 {letter}: {len(self.data[letter])} samples")
    
    def get_landmarks(self, frame):
        """Extract landmarks from frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)
        
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand = results.hand_landmarks[0]
            coords = []
            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords), hand
        return None, None
    
    def draw_hand(self, frame, hand_landmarks):
        """Draw hand on frame."""
        h, w = frame.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        
        # Lines
        for conn in HAND_CONNECTIONS:
            cv2.line(frame, points[conn[0]], points[conn[1]], (0, 255, 0), 2)
        
        # Points
        for pt in points:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
    
    def run(self):
        """Main recording loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*50)
        print("🎥 NGT LETTER RECORDER")
        print("="*50)
        print("SPACE - save")
        print("1,2,3,4 - switch letter (H,P,T,W)")
        print("Q - quit")
        print("="*50 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror
            current_letter = LETTERS[self.current_letter_idx]
            
            # Hand detection
            landmarks, hand = self.get_landmarks(frame)
            hand_detected = landmarks is not None
            
            if hand_detected:
                self.draw_hand(frame, hand)
            
            # UI
            color = (0, 255, 0) if hand_detected else (0, 0, 255)
            status = "HAND OK" if hand_detected else "NO HAND"
            
            # Header
            cv2.rectangle(frame, (0, 0), (640, 100), (40, 40, 40), -1)
            cv2.putText(frame, f"Letter: {current_letter}", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(frame, f"Count: {len(self.data[current_letter])}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, status, (300, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Bottom hints
            y = 460
            for i, letter in enumerate(LETTERS):
                cnt = len(self.data[letter])
                clr = (0, 255, 0) if cnt >= 50 else (0, 255, 255) if cnt > 0 else (128, 128, 128)
                txt = f"{i+1}:{letter}({cnt})"
                cv2.putText(frame, txt, (10 + i*150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)
            
            cv2.imshow("NGT Recorder", frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space - save
                if hand_detected:
                    # Protection against spam (min 0.2 sec between records)
                    if time.time() - self.last_save_time > 0.2:
                        self.data[current_letter].append(landmarks)
                        self.last_save_time = time.time()
                        print(f"✅ {current_letter}: {len(self.data[current_letter])}")
            elif key == ord('1'):
                self.current_letter_idx = 0
            elif key == ord('2'):
                self.current_letter_idx = 1
            elif key == ord('3'):
                self.current_letter_idx = 2
            elif key == ord('4'):
                self.current_letter_idx = 3
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Saving
        print("\n💾 Saving...")
        self.save_all()
        print("✅ Done!")
        
        # Summary
        print("\n📊 SUMMARY:")
        for letter in LETTERS:
            cnt = len(self.data[letter])
            status = "✅" if cnt >= 50 else "⚠️"
            print(f"   {status} {letter}: {cnt} samples")
 
 
if __name__ == "__main__":
    recorder = LandmarkRecorder()
    recorder.run()