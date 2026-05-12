# Sign Language Learning App

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

A real-time, interactive sign language learning application powered by computer vision and deep learning. Learn the NGT (Nederlandse Gebarentaal / Dutch Sign Language) alphabet with instant feedback from AI models.

## Features

- **Real-time Hand Detection** - Uses MediaPipe for accurate hand landmark detection
- **Dual AI Models** - CNN for static letters (A-I, K-Y) and LSTM for dynamic letters (J, Z)
- **Live Feedback** - Instant recognition and validation of your sign language gestures
- **Two Learning Modes** - Sequential (ABC order) or Random letter practice
- **Bilingual Support** - Interface available in English and Dutch (Nederlands)
- **Recording-Based Workflow** - 3-second recording window for accurate gesture capture
- **Smart Hints System** - Get helpful guidance when struggling with a letter
- **Progress Tracking** - Monitor your accuracy, attempts, and completed letters
- **Tutorial GIFs** - Visual examples for each letter
- **Skip Function** - Move to the next letter at any time

## Architecture

### Deep Learning Models

1. **Static Sign Detector (CNN)**
   - **Architecture**: ResidualMLP with skip connections
   - **Input**: 63 features (21 hand landmarks × 3 coordinates)
   - **Output**: 24 classes (A-I, K-Y, excluding J and Z)
   - **Accuracy**: ~95% on validation set
   - **Location**: `models/static/best_model.pth`

2. **Dynamic Sign Detector (LSTM)**
   - **Architecture**: Bidirectional LSTM with attention
   - **Input**: Sequences of 30 frames × 63 features
   - **Output**: 2 classes (J, Z - letters requiring movement)
   - **Accuracy**: ~92% on validation set
   - **Location**: `models/dynamic/best_model.pth`

### Backend Stack

- **FastAPI** - High-performance async web framework
- **WebSocket** - Real-time bidirectional communication
- **MediaPipe** - Hand landmark detection and tracking
- **PyTorch** - Deep learning inference
- **OpenCV** - Image processing and video capture

### Frontend Stack

- **Vanilla JavaScript** - No frameworks, pure performance
- **WebRTC** - Browser-based webcam access
- **WebSocket API** - Real-time data streaming
- **CSS3** - Modern, responsive UI design

## Prerequisites

- Python 3.12 or higher (required for modern type hints and performance)
- Webcam (for hand gesture capture)
- GPU (Optional) - CUDA-compatible GPU for faster inference

## Installation

### Option 1: Using uv (Recommended - Fast)

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver, written in Rust.

1. **Install uv** (if not already installed):
   ```bash
   # On Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/Gfgf96/DSL-Learning.git
   cd sign-language
   ```

3. **Install dependencies and create virtual environment**:
   ```bash
   uv sync
   ```

4. **Activate the virtual environment**:
   ```bash
   # On Windows
   .venv\Scripts\activate
   
   ```bash
   # On macOS/Linux
   source .venv/bin/activate
   ```

> [!NOTE]
> **Model Auto-Download:** The deep learning models and MediaPipe trackers are large files and are not included in the repository. They will be **automatically downloaded** from GitHub Releases the first time you run the application.

## Usage

### Starting the Application

1. **Ensure you're in the project directory with activated virtual environment**

2. **Run the application**:
   ```bash
   python main.py
   ```

   This now downloads missing models and builds the frontend export automatically on startup, so the FastAPI app serves the web UI at `http://localhost:8000/`.

3. **Open your browser**:
   - Navigate to `http://localhost:8000/`
   - The app will automatically use your webcam

### Using the App

1. **Click "Start Camera"** - Allow browser access to your webcam
2. **Choose your learning mode**:
   - **ABC** - Practice letters in alphabetical order
   - **Random** - Practice letters in random order
3. **Position your hand** - Ensure your hand is clearly visible
4. **Click "Record"** - Hold the sign for 3 seconds
5. **Get instant feedback** - The AI validates your gesture
6. **Progress through the alphabet** - Complete all letters or use "Skip" to move on

### Keyboard Shortcuts & Controls

- **Start Camera** - Begin the session
- **Record** - Capture a 3-second gesture recording
- **Skip Letter** - Move to the next letter immediately
- **Mode Toggle** - Switch between ABC and Random order
- **Language Toggle** - Switch between English and Nederlands

## Project Structure

```
sign-language/
├── train/                      # Training and dataset utilities
│   ├── CNN_model/              # Static sign classifier (ResidualMLP)
│   │   ├── model.py            # CNN architecture
│   │   ├── train.py            # Training script
│   │   ├── config.py           # Model configuration
│   │   └── inference.py        # Inference utilities
│   ├── LSTM_model/             # Dynamic sign classifier (LSTM)
│   │   ├── model.py            # LSTM architecture
│   │   ├── train.py            # Training script
│   │   └── config.py           # Model configuration
│   ├── data_collect/           # Data collection utilities
│   └── dataset_builder/        # Dataset creation tools
├── models/                     # Trained model weights
│   ├── static/                 # CNN model (A-I, K-Y)
│   │   ├── best_model.pth
│   │   └── classes.npy
│   ├── dynamic/                # LSTM model (J, Z)
│   │   ├── best_model.pth
│   │   └── classes.npy
│   └── mediapipe/              # Shared mediapipe models
│       └── hand_landmarker.task
├── src/
│   ├── assets/                 # Tutorial GIFs
│   ├── cli.py                  # CLI application entry point
│   └── backend/
│       ├── api/                # FastAPI routes and schemas
│       │   ├── routes.py       # WebSocket & REST endpoints
│       │   └── schemas.py      # Pydantic models
│       ├── core/               # Core business logic
│       │   ├── config.py       # App configuration
│       │   ├── session_manager.py   # User session handling
│       │   ├── letter_sequence.py   # Letter progression
│       │   └── tutorial_manager.py  # Tutorial system
│       ├── detection/          # Hand detection & prediction
│       │   ├── hand_capture.py      # MediaPipe integration
│       │   ├── static_detector.py   # CNN predictor
│       │   ├── dynamic_detector.py  # LSTM predictor
│       │   └── sign_detector.py     # Unified detector
│       └── models/             # Model architectures
│           ├── cnn_model.py    # ResidualMLP
│           ├── downloader.py   # Auto-downloader script
│           └── config.py       # Model config
├── frontend/                   # Web interface
│   ├── index.html              # Main HTML structure
│   ├── app.js                  # Application logic
│   └── app.css                 # Styling
├── main.py                     # Web API entry point
├── cli.py                      # Root CLI runner
├── pyproject.toml            # Python project metadata (uv)
└── README.md                 # This file
```

## Configuration

### Application Settings

Edit `src/backend/core/config.py` to customize:

```python
# Detection settings
RECORDING_DURATION = 3.0        # Seconds per recording
CONFIDENCE_THRESHOLD = 0.6      # Minimum confidence for valid prediction

# Session settings
MAX_ATTEMPT_TIME = 60           # Timeout per letter (seconds)
HINT_THRESHOLD_ATTEMPTS = 3     # Attempts before showing hint
MAX_HINTS = 2                   # Maximum hints per letter

# Dynamic letter settings
DYNAMIC_BUFFER_SIZE = 30        # Frames needed for LSTM (J, Z)
```

### Model Configuration

#### CNN Model (`src/backend/models/config.py`):
```python
INPUT_SIZE = 63                 # 21 landmarks × 3 coordinates
NUM_CLASSES = 25                # 24 letters + nonsense class
BATCH_SIZE = 64
LEARNING_RATE = 0.001
```

#### LSTM Model (`LSTM_model/config.py`):
```python
SEQUENCE_LENGTH = 30            # Frames per sequence
INPUT_SIZE = 63                 # Features per frame
HIDDEN_SIZE = 128               # LSTM hidden units
NUM_LAYERS = 2                  # LSTM layers
```

## How It Works

### 1. Hand Detection
- **MediaPipe Hands** detects 21 hand landmarks in real-time
- Each landmark has 3D coordinates (x, y, z)
- Landmarks are normalized (centered on wrist, scaled by hand size)

### 2. Feature Extraction
- 21 landmarks × 3 coordinates = **63 features** per frame
- For static letters: Single frame processed by CNN
- For dynamic letters (J, Z): 30-frame sequence processed by LSTM

### 3. Sign Classification

**Static Letters (A-I, K-Y):**
1. User clicks "Record"
2. System captures frames for 3 seconds
3. Each frame processed by CNN
4. Majority vote determines predicted letter
5. Validation against target letter

**Dynamic Letters (J, Z):**
1. User clicks "Record"
2. System collects 30 consecutive frames
3. Sequence processed by LSTM
4. Single prediction for the movement
5. Validation against target letter

### 4. Feedback Loop
- **Match**: High confidence + correct letter → Success! Move to next letter
- **Mismatch**: Wrong letter → Try again, hints after 3 attempts
- **Low Confidence**: Below threshold → "Not recognized" feedback
- **Timeout**: 60 seconds elapsed - Auto-skip to next letter

## Training Your Own Models

### Static Model (CNN)

1. **Collect landmark data** (see `data_collect/record_landmarks.py`)
2. **Prepare dataset** in `.npz` format with normalized landmarks
3. **Train the model**:
   ```bash
   cd CNN_model
   python train.py
   ```
4. **Model saved to**: `models/static/best_model.pth`

### Dynamic Model (LSTM)

1. **Collect sequence data** for J and Z gestures
2. **Format as**: `(num_samples, 30, 63)` numpy array
3. **Train the model**:
   ```bash
   cd LSTM_model
   python train.py
   ```
4. **Model saved to**: `models/dynamic/best_model.pth`

### Data Format

**Static letters** (`ngt.npz`):
```python
{
    'X': np.array((N, 63)),      # N samples of 63 features
    'y': np.array((N,)),         # Letter labels
    'classes': np.array((24,))   # Class names
}
```

**Dynamic letters** (`jz_dynamic.npz`):
```python
{
    'X': np.array((N, 30, 63)),  # N sequences of 30 frames
    'y': np.array((N,))          # Labels: 'J' or 'Z'
}
```

## Troubleshooting

### Common Issues

**Webcam not working:**
- Ensure browser has webcam permissions
- Check if another app is using the webcam
- Try a different browser (Chrome/Edge recommended)

**Models not loading:**
- Verify `models/static/best_model.pth` exists
- Verify `models/dynamic/best_model.pth` exists
- Check file paths in `src/backend/core/config.py`

**Poor recognition accuracy:**
- Ensure good lighting conditions
- Position hand clearly in frame (centered, proper distance)
- Hold gesture steady during 3-second recording
- Match hand orientation to tutorial GIF

**Port already in use:**
```bash
# Change port in main.py or kill existing process:
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

**Dependencies issues:**
```bash
# Reinstall/sync all dependencies with uv
uv sync
```

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/sign-language.git
cd sign-language

# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests (if available)
pytest

# Format code
black src/
isort src/
```

## Performance Metrics

| Model | Accuracy | F1-Score | Inference Time |
|-------|----------|----------|----------------|
| CNN (Static) | 94.8% | 0.947 | ~5ms per frame |
| LSTM (Dynamic) | 91.5% | 0.912 | ~15ms per sequence |

**Hardware**: Intel i7-12700K, RTX 3070, 32GB RAM

## Roadmap

- Add more sign languages (ASL, BSL, etc.)
- Word and sentence recognition
- Mobile app (iOS/Android)
- Multiplayer practice mode
- Gamification with achievements
- Export progress reports
- Offline mode support
- Voice feedback option

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Oleksii Krasnoshtanov** - [@OleksiiKrasnoshtanov240247](https://github.com/OleksiiKrasnoshtanov240247)

## Acknowledgments

- **MediaPipe** team for hand tracking technology
- **PyTorch** community for deep learning framework
- **FastAPI** for excellent async web framework
- **NGT community** for sign language resources
- All contributors and testers

## Support

- **Issues**: [GitHub Issues](https://github.com/Gfgf96/DSL-Learning/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Gfgf96/DSL-Learning/discussions)
- **Email**: Contact via GitHub profile

## Star the Project

If this project helped you learn sign language, please consider giving it a star on GitHub.

---

**Made with care for the deaf and hard-of-hearing community**
