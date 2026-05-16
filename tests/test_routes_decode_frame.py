import base64
import cv2
import numpy as np
from src.backend.api.routes import decode_frame


def test_decode_frame_from_base64():
    # Create a small image and encode to jpg
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[0, 0] = [255, 0, 0]
    success, buf = cv2.imencode(".jpg", img)
    assert success
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    frame = decode_frame(b64)
    assert frame is not None
    assert frame.shape[0] == 10 and frame.shape[1] == 10
