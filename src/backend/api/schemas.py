"""
Pydantic schemas for API request/response validation.
"""
from typing import Optional, Dict
from pydantic import BaseModel


class PredictionResult(BaseModel):
    """Prediction result from model."""
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]


class FrameData(BaseModel):
    """Frame data sent from client."""
    frame: str  # Base64 encoded image
    session_id: Optional[str] = None


class DetectionResponse(BaseModel):
    """Response sent to client after processing frame."""
    hand_detected: bool
    prediction: Optional[PredictionResult] = None
    match: bool = False
    success: bool = False
    timeout: bool = False
    show_hint: bool = False
    hint_message: str = ""
    tutorial_url: Optional[str] = None
    progress: Dict
    current_letter: Optional[str] = None
    consecutive_matches: int
    matches_needed: int


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    current_letter: Optional[str] = None
    total_correct: int
    total_attempts: int
    accuracy: float
    completed_letters: list
    mode: str = "sequential"  # "sequential", "random", or "sentence"
    target_sentence: str = ""
    recognized_sentence: str = ""

class ModeChangeRequest(BaseModel):
    """Request to change letter sequence mode."""
    mode: str  # "sequential", "random", or "sentence"


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
