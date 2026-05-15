"""
Webcam test for LSTM Dynamic Sign Model (J/Z)
==============================================
Controls:
  SPACE - Start recording gesture
  Q - Quit
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from LSTM_model.inference import DynamicSignPredictor

# MediaPipe model path
MODEL_PATH = SCRIPT_DIR.parent.parent / "models" / "mediapipe" / "hand_landmarker.task"

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


class WebcamTester:
    def __init__(self):
        # MediaPipe hand detector
        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=1, min_hand_detection_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # LSTM predictor
        self.predictor = DynamicSignPredictor()

        # State
        self.last_prediction = None

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
        """Draw hand skeleton."""
        h, w = frame.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for conn in HAND_CONNECTIONS:
            cv2.line(frame, points[conn[0]], points[conn[1]], (0, 255, 0), 2)
        for pt in points:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    def draw_progress_bar(self, frame, progress):
        """Draw recording progress bar."""
        bar_width = 300
        bar_height = 30
        x, y = 170, 420

        # Background
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)

        # Progress
        fill_width = int(bar_width * progress)
        color = (0, 255, 255) if progress < 1.0 else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + fill_width, y + bar_height), color, -1)

        # Border
        cv2.rectangle(
            frame, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 2
        )

        # Text
        pct = int(progress * 100)
        cv2.putText(
            frame,
            f"{pct}%",
            (x + bar_width + 10, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    def run(self):
        """Main loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n" + "=" * 50)
        print("🎥 LSTM WEBCAM TEST (J/Z)")
        print("=" * 50)
        print("SPACE - Start recording gesture")
        print("Q - Quit")
        print("=" * 50 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Detect hand
            landmarks, hand = self.get_landmarks(frame)
            hand_detected = landmarks is not None

            if hand_detected:
                self.draw_hand(frame, hand)

                # If collecting, add frame
                if self.predictor.is_collecting:
                    result = self.predictor.add_frame(landmarks)
                    if result:
                        self.last_prediction = result
                        self.predictor.is_collecting = False

            # UI - Header
            cv2.rectangle(frame, (0, 0), (640, 80), (40, 40, 40), -1)

            # Status
            if self.predictor.is_collecting:
                status = "RECORDING..."
                color = (0, 255, 255)
            elif hand_detected:
                status = "HAND OK - Press SPACE"
                color = (0, 255, 0)
            else:
                status = "NO HAND"
                color = (0, 0, 255)

            cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Last prediction
            if self.last_prediction:
                letter = self.last_prediction["letter"]
                conf = self.last_prediction["confidence"]
                cv2.putText(
                    frame,
                    f"Prediction: {letter} ({conf:.2f})",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

            # Progress bar if recording
            if self.predictor.is_collecting:
                progress = self.predictor.get_buffer_progress()
                self.draw_progress_bar(frame, progress)

            # Instructions
            cv2.putText(
                frame,
                "SPACE: Record | Q: Quit",
                (10, 475),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow("LSTM Test (J/Z)", frame)

            # Controls
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                if hand_detected and not self.predictor.is_collecting:
                    print("[INFO] Recording started...")
                    self.predictor.start_collecting()
                    self.last_prediction = None

        cap.release()
        cv2.destroyAllWindows()
        print("👋 Done!")


if __name__ == "__main__":
    tester = WebcamTester()
    tester.run()
