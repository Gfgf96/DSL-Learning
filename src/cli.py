"""
Command-line interface for sign language detector.
Refactored version using new modular architecture.

Usage:
    python -m src.cli_new
    or
    python src/cli_new.py
"""
import cv2
import time
import sys
import torch
from pathlib import Path

# Import from new restructured modules
from src.backend.detection import SignDetector


def run_app():
    """Main application loop for real-time sign language detection."""
    print("=== Sign Language Detector (Refactored) ===")
    print("Initializing...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Check and download models
    from src.backend.models.downloader import check_and_download_models
    check_and_download_models()
    
    # Path to models
    STATIC_MODEL_PATH = Path(__file__).parent.parent / "models" / "static" / "best_model.pth"
    DYNAMIC_MODEL_PATH = Path(__file__).parent.parent / "models" / "dynamic" / "best_model.pth"
    
    # Check if models exist
    static_ckpt = str(STATIC_MODEL_PATH) if STATIC_MODEL_PATH.exists() else None
    dynamic_ckpt = str(DYNAMIC_MODEL_PATH) if DYNAMIC_MODEL_PATH.exists() else None
    
    if static_ckpt:
        print(f"Found static model at: {static_ckpt}")
    else:
        print(f"Warning: Static model not found at {STATIC_MODEL_PATH}")
        
    if dynamic_ckpt:
        print(f"Found dynamic model at: {dynamic_ckpt}")
    else:
        print(f"Warning: Dynamic model not found at {DYNAMIC_MODEL_PATH}")
    
    if not static_ckpt and not dynamic_ckpt:
        print("Detector will run without prediction.")
    
    # Initialize detector
    try:
        detector = SignDetector(static_model_path=static_ckpt, dynamic_model_path=dynamic_ckpt, device=device)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
        
    target_letter = 'A'
    
    # FPS Control
    TARGET_FPS = 15
    interval = 1.0 / TARGET_FPS
    last_capture = 0
    
    print("\nControls:")
    print("  A-Z: Change target letter")
    print("  Q: Quit")
    print("\nStarting detection...\n")
    
    try:
        while True:
            # FPS throttling
            now = time.time()
            if now - last_capture < interval:
                time.sleep(0.001)
                continue
            last_capture = now
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame through detector
            landmarks, status, data, prediction_info = detector.process_frame(
                frame.copy(), 
                target_letter
            )
            
            # Visualize hand landmarks
            detector.hand_capture.visualize_landmarks(frame)
            
            # Display status
            color = (0, 255, 0) if "Hand: YES" in status else (0, 0, 255)
            if "[MATCH]" in status:
                color = (0, 255, 0)  # Green for match
            
            cv2.putText(frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"FPS Limit: {TARGET_FPS} | Target: {target_letter}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, "Press 'Q' to quit | A-Z to change letter", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Display top predictions
            if prediction_info and prediction_info.get('all_probabilities'):
                sorted_probs = sorted(
                    prediction_info['all_probabilities'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                
                y_offset = 115
                for letter, prob in sorted_probs:
                    text_color = (0, 255, 255)  # Yellow
                    if letter == target_letter:
                        text_color = (0, 255, 0)  # Green for target
                        
                    cv2.putText(frame, f"{letter}: {prob:.2f}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    y_offset += 20
            
            # Show frame
            cv2.imshow("Sign Language Detector", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif ord('a') <= key <= ord('z'):
                target_letter = chr(key).upper()
                detector.clear_buffer()
                print(f"Target letter changed to: {target_letter}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Detector closed.")


if __name__ == "__main__":
    run_app()
