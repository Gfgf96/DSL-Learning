"""
Hand landmark detection using MediaPipe.
Migrated from src/backend/video/capture.py
"""
import cv2
import mediapipe as mp
import numpy as np


class HandCapture:
    """Captures and processes hand landmarks using MediaPipe."""
    
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.results = None

    def extract_landmarks(self, frame):
        """
        Extract hand landmarks from a frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            landmarks: np.array of shape (21, 3) or None if no hand detected
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks:
            landmarks = self.results.multi_hand_landmarks[0].landmark
            coordinates = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
            return coordinates

        return None

    def visualize_landmarks(self, frame):
        """
        Draw landmarks on frame for visualization.
        
        Args:
            frame: BGR image to draw on (modified in-place)
        """
        if self.results and self.results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                self.results.multi_hand_landmarks[0],
                self.mp_hands.HAND_CONNECTIONS
            )

    def run_capture_loop(self):
        """
        Main capture loop - opens camera and displays feed.
        Useful for debugging/testing hand detection.
        """
        webcam = cv2.VideoCapture(0)

        if not webcam.isOpened():
            print("Error: Could not open webcam")
            return

        print("Press 'q' to quit")

        while True:
            ret, frame = webcam.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Extract landmarks
            landmarks = self.extract_landmarks(frame)

            # Visualize landmarks on frame
            self.visualize_landmarks(frame)

            # Display landmarks info
            if landmarks is not None:
                cv2.putText(frame, f"Hand detected: {landmarks.shape}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No hand detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display frame
            cv2.imshow('Hand Tracking', frame)

            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        webcam.release()
        cv2.destroyAllWindows()


def normalize(landmarks):
    """
    Normalize landmarks by centering on wrist and scaling by hand size.
    
    Args:
        landmarks: np.array of shape (21, 3) - raw hand landmarks
        
    Returns:
        normalized: np.array of shape (21, 3) - normalized landmarks
    """
    lm = landmarks.copy()
    lm -= lm[0]  # center on wrist (landmark 0)
    
    # Scale by distance to middle finger tip (landmark 12)
    scale = np.linalg.norm(lm[12])  
    
    if scale > 0:
        lm /= scale
        
    return lm


if __name__ == "__main__":
    capture = HandCapture()
    capture.run_capture_loop()
