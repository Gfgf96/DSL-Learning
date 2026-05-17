"""
FastAPI routes for sign language detection API - Recording-based workflow.
"""

import asyncio
import base64
import cv2
import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional

from src.backend.api.schemas import SessionInfo, ModeChangeRequest
from src.backend.core.session_manager import SessionManager
from src.backend.core.config import (
    STATIC_MODEL_PATH,
    DYNAMIC_MODEL_PATH,
    DYNAMIC_LETTERS,
)
from src.backend.detection.hand_capture import HandCapture, normalize
from src.backend.detection.static_detector import StaticSignPredictor
from src.backend.detection.dynamic_detector import DynamicSignPredictor

# Add CORS middleware will be configured on app below

# Global instances
session_manager = SessionManager()
static_predictor = None
dynamic_predictor = None
hand_capture = None

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_OUT_DIR = FRONTEND_DIR / "out"
ASSETS_DIR = PROJECT_ROOT / "src" / "assets"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager to initialize and cleanup global predictors."""
    global static_predictor, dynamic_predictor, hand_capture

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Starting server on {device}")

    # Initialize hand capture
    hand_capture = HandCapture()
    print(" Hand capture initialized")

    # Initialize static model (CNN)
    if STATIC_MODEL_PATH.exists():
        try:
            static_predictor = StaticSignPredictor(
                str(STATIC_MODEL_PATH), device=device
            )
            print(f" Static model loaded: {STATIC_MODEL_PATH.name}")
        except Exception as e:
            print(f" Failed to load static model: {e}")
    else:
        print(f" Static model not found: {STATIC_MODEL_PATH}")

    # Initialize dynamic model (LSTM)
    if DYNAMIC_MODEL_PATH.exists():
        try:
            dynamic_predictor = DynamicSignPredictor(
                str(DYNAMIC_MODEL_PATH), device=device
            )
            print(f" Dynamic model loaded: {DYNAMIC_MODEL_PATH.name}")
        except Exception as e:
            print(f" Failed to load dynamic model: {e}")
    else:
        print(f" Dynamic model not found: {DYNAMIC_MODEL_PATH}")

    yield


# Create the FastAPI app with lifespan support
app = FastAPI(title="Sign Language Learning API", lifespan=lifespan)

# Configure CORS middleware (must be before app starts)
# NOTE: For demo purposes we allow all origins. In production restrict
# `allow_origins` to the known frontend host(s) and avoid `"*"`.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# get_index removed in favor of root static mount


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "static_model": static_predictor is not None,
        "dynamic_model": dynamic_predictor is not None,
        "hand_capture": hand_capture is not None,
    }


@app.post("/api/session/new", response_model=SessionInfo)
async def create_session(mode: str = "sequential"):
    """Create a new learning session."""
    session = session_manager.create_session(mode=mode)
    progress = session.get_progress()

    return SessionInfo(
        session_id=session.id,
        current_letter=session.current_letter,
        total_correct=session.total_correct,
        total_attempts=session.total_attempts,
        accuracy=progress["accuracy"],
        completed_letters=session.completed_letters,
        mode=session.mode,
        target_sentence=session.target_sentence,
        recognized_sentence=session.recognized_sentence,
    )


@app.get("/api/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get session information."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    progress = session.get_progress()
    return SessionInfo(
        session_id=session.id,
        current_letter=session.current_letter,
        total_correct=session.total_correct,
        total_attempts=session.total_attempts,
        accuracy=progress["accuracy"],
        completed_letters=session.completed_letters,
        mode=session.mode,
        target_sentence=session.target_sentence,
        recognized_sentence=session.recognized_sentence,
    )


@app.post("/api/session/{session_id}/mode")
async def change_mode(session_id: str, request: ModeChangeRequest):
    """Change letter sequence mode for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        session.set_mode(request.mode)
        return {
            "status": "success",
            "mode": session.mode,
            "message": f"Mode changed to {request.mode}",
            "progress": session.get_progress(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    session_manager.remove_session(session_id)
    return {"status": "deleted"}


def decode_frame(frame_base64: str) -> np.ndarray:
    """Decode base64 frame to numpy array."""
    # Remove data URL prefix if present
    if "," in frame_base64:
        frame_base64 = frame_base64.split(",")[1]

    # Decode base64
    img_bytes = base64.b64decode(frame_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return frame


def is_dynamic_letter(letter: Optional[str]) -> bool:
    """Check if letter requires dynamic detection (LSTM)."""
    if not letter:
        return False
    return letter.upper() in DYNAMIC_LETTERS


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time sign language detection.

    New workflow:
    1. Client connects and gets session
    2. Show target letter and GIF
    3. Client starts camera (preview only)
    4. User clicks "Record" button
    5. Server receives 'start_recording' command
    6. Collect frames for 3-5 seconds
    7. Server processes recording and returns result
    8. Show success/failure, move to next letter

    Messages:
    - Client -> Server: {"type": "start_recording", "session_id": "..."}
    - Client -> Server: {"type": "frame", "frame": "base64...", "session_id": "..."}
    - Client -> Server: {"type": "stop_recording", "session_id": "..."}
    - Server -> Client: DetectionResponse
    """
    await websocket.accept()
    print("[OK] WebSocket connected")

    session = None
    is_dynamic = False

    try:
        while True:
            # Receive message from client
            try:
                data = await websocket.receive_json()
            except Exception as e:
                print(f"❌ Error receiving message: {e}")
                break

            message_type = data.get("type", "frame")

            # Get or create session
            session_id = data.get("session_id")
            if not session:
                if session_id:
                    session = session_manager.get_session(session_id)
                if not session:
                    session = session_manager.create_session(session_id)
                    print(f"📝 New session: {session.id}")

                # Check if current letter is dynamic
                is_dynamic = is_dynamic_letter(session.current_letter)

            # Handle different message types
            if message_type == "start_recording":
                # Start recording
                session.start_recording()

                # If dynamic, start collecting frames
                if is_dynamic and dynamic_predictor:
                    dynamic_predictor.start_collecting()

                response = {
                    "session_id": session.id,
                    "recording": True,
                    "current_letter": session.current_letter,
                    "is_dynamic": is_dynamic,
                    "message": "Recording started",
                }
                await websocket.send_json(response)
                continue

            elif message_type == "stop_recording":
                # Force stop recording
                if session.is_recording:
                    result = session.finish_recording()

                    if is_dynamic and dynamic_predictor:
                        dynamic_predictor.stop_collecting()

                    response = {
                        "session_id": session.id,
                        "recording": False,
                        **result,
                        "progress": session.get_progress(),
                    }
                    await websocket.send_json(response)
                continue

            elif message_type == "set_sentence":
                target_sentence = data.get("sentence", "")
                session.set_mode("sentence")
                session.set_target_sentence(target_sentence)

                response = {
                    "session_id": session.id,
                    "mode": session.mode,
                    "target_sentence": session.target_sentence,
                    "recognized_sentence": session.recognized_sentence,
                    "message": "Target sentence set"
                    if target_sentence
                    else "Free sign mode activated",
                    "progress": session.get_progress(),
                }
                await websocket.send_json(response)
                continue

            elif message_type == "clear_sentence":
                session.clear_recognized()
                response = {
                    "session_id": session.id,
                    "recognized_sentence": session.recognized_sentence,
                    "message": "Recognized sentence cleared",
                    "progress": session.get_progress(),
                }
                await websocket.send_json(response)
                continue

            elif message_type == "skip":
                # Skip current letter and move to next one
                print(f"[INFO] Skipping letter: {session.current_letter}")

                # Stop recording if active
                if session.is_recording:
                    session.is_recording = False
                    if is_dynamic and dynamic_predictor:
                        dynamic_predictor.stop_collecting()
                        dynamic_predictor.clear_buffer()

                # Skip to next letter
                result = session.skip_letter()

                # Update is_dynamic flag for new letter
                is_dynamic = is_dynamic_letter(session.current_letter)

                # Send response with new letter - same structure as timeout/success
                response = {
                    "session_id": session.id,
                    "hand_detected": False,
                    "recording": False,
                    "match": result.get("match", False),
                    "success": result.get("success", False),
                    "timeout": result.get("timeout", False),
                    "skipped": result.get("skipped", True),
                    "show_hint": result.get("show_hint", False),
                    "hint_message": result.get("hint_message", ""),
                    "message": result.get("message", "Letter skipped"),
                    "progress": session.get_progress(),
                }
                await websocket.send_json(response)
                continue

            elif message_type == "frame":
                # Process frame
                frame_base64 = data.get("frame")
                if not frame_base64:
                    continue

                # Decode frame
                try:
                    frame = await asyncio.to_thread(decode_frame, frame_base64)
                except Exception as e:
                    print(f" Frame decode error: {e}")
                    continue

                # Extract landmarks
                landmarks = await asyncio.to_thread(
                    hand_capture.extract_landmarks, frame
                )

                if landmarks is None:
                    # No hand detected
                    if session.is_recording:
                        # During recording, add a "no hand" entry
                        response = {
                            "session_id": session.id,
                            "hand_detected": False,
                            "recording": True,
                            "message": "No hand detected",
                            "progress": session.get_progress(),
                        }
                    else:
                        # Just preview
                        response = {
                            "session_id": session.id,
                            "hand_detected": False,
                            "recording": False,
                            "current_letter": session.current_letter,
                            "progress": session.get_progress(),
                        }

                    await websocket.send_json(response)
                    continue

                # Hand detected
                norm_landmarks = normalize(landmarks)

                # If not recording, just send preview status
                if not session.is_recording:
                    response = {
                        "session_id": session.id,
                        "hand_detected": True,
                        "recording": False,
                        "current_letter": session.current_letter,
                        "message": "Hand detected - Click Record to start",
                        "progress": session.get_progress(),
                    }
                    await websocket.send_json(response)
                    continue

                # Recording in progress
                if is_dynamic:
                    # Dynamic letter (J, Z) - collect frames for LSTM
                    if dynamic_predictor:
                        is_ready = dynamic_predictor.add_frame(landmarks)

                        if is_ready:
                            # Buffer full, make prediction
                            try:
                                pred_result = await asyncio.to_thread(
                                    dynamic_predictor.predict
                                )
                            except Exception as e:
                                print(f"❌ Dynamic prediction error: {e}")
                                pred_result = None

                            if pred_result:
                                if pred_result["predicted_class"] == "Nonsense":
                                    pred_result["predicted_class"] = "-"

                                # Add to session
                                result = session.add_prediction(
                                    pred_result["predicted_class"],
                                    pred_result["confidence"],
                                )

                                # Finish recording if not already finished
                                if not result:
                                    result = session.finish_recording()

                                dynamic_predictor.clear_buffer()
                                is_dynamic = is_dynamic_letter(session.current_letter)

                                response = {
                                    "session_id": session.id,
                                    "hand_detected": True,
                                    "recording": False,
                                    "prediction": {
                                        "predicted_class": pred_result[
                                            "predicted_class"
                                        ],
                                        "confidence": pred_result["confidence"],
                                    },
                                    "match": result.get("match", False),
                                    "success": result.get("success", False),
                                    "timeout": result.get("timeout", False),
                                    "message": result.get("message", ""),
                                    "show_hint": result.get("show_hint", False),
                                    "hint_message": result.get("hint_message", ""),
                                    "progress": session.get_progress(),
                                }
                                await websocket.send_json(response)
                        else:
                            # Still collecting
                            progress_pct = dynamic_predictor.get_buffer_progress()
                            response = {
                                "session_id": session.id,
                                "hand_detected": True,
                                "recording": True,
                                "buffer_progress": progress_pct,
                                "message": f"Collecting frames... {int(progress_pct * 100)}%",
                                "progress": session.get_progress(),
                            }
                            await websocket.send_json(response)
                    else:
                        # No dynamic model
                        response = {
                            "session_id": session.id,
                            "hand_detected": True,
                            "recording": False,
                            "message": "Dynamic model not available",
                            "progress": session.get_progress(),
                        }
                        await websocket.send_json(response)
                        session.is_recording = False

                else:
                    # Static letter - use CNN
                    if static_predictor:
                        try:
                            pred_result = await asyncio.to_thread(
                                static_predictor.predict, norm_landmarks
                            )
                        except Exception as e:
                            print(f"❌ Static prediction error: {e}")
                            continue

                        if pred_result["predicted_class"] == "Nonsense":
                            pred_result["predicted_class"] = "-"

                        # Add prediction to session
                        result = session.add_prediction(
                            pred_result["predicted_class"], pred_result["confidence"]
                        )

                        if result:
                            # Recording finished
                            is_dynamic = is_dynamic_letter(session.current_letter)

                            response = {
                                "session_id": session.id,
                                "hand_detected": True,
                                "recording": False,
                                "prediction": {
                                    "predicted_class": pred_result["predicted_class"],
                                    "confidence": pred_result["confidence"],
                                },
                                "match": result.get("match", False),
                                "success": result.get("success", False),
                                "timeout": result.get("timeout", False),
                                "message": result.get("message", ""),
                                "show_hint": result.get("show_hint", False),
                                "hint_message": result.get("hint_message", ""),
                                "progress": session.get_progress(),
                            }
                            await websocket.send_json(response)
                        else:
                            # Still recording
                            response = {
                                "session_id": session.id,
                                "hand_detected": True,
                                "recording": True,
                                "prediction": {
                                    "predicted_class": pred_result["predicted_class"],
                                    "confidence": pred_result["confidence"],
                                },
                                "message": "Recording in progress...",
                                "progress": session.get_progress(),
                            }
                            await websocket.send_json(response)
                    else:
                        # No static model
                        response = {
                            "session_id": session.id,
                            "hand_detected": True,
                            "recording": False,
                            "message": "Static model not available",
                            "progress": session.get_progress(),
                        }
                        await websocket.send_json(response)
                        session.is_recording = False

    except WebSocketDisconnect:
        print("📡 WebSocket disconnected normally")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback

        traceback.print_exc()
        try:
            error_response = {
                "error": str(e),
                "message": "Server error occurred",
                "recording": False,
            }
            await websocket.send_json(error_response)
        except Exception as send_error:
            print(f"Failed to send error message: {send_error}")


# Mount static files after routes to avoid conflicts
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# Mount Next.js _next static files if they exist
if (FRONTEND_OUT_DIR / "_next").exists():
    app.mount(
        "/_next",
        StaticFiles(directory=str(FRONTEND_OUT_DIR / "_next")),
        name="next_static",
    )

# Mount Next.js out directory for static files (images, index.html, etc) at root
if FRONTEND_OUT_DIR.exists() and FRONTEND_OUT_DIR.is_dir():
    app.mount(
        "/", StaticFiles(directory=str(FRONTEND_OUT_DIR), html=True), name="static_out"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
