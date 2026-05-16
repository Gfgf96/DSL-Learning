from src.backend.core.tutorial_manager import TutorialManager


def test_scan_available_gifs(tmp_path):
    # Create a fake assets dir with a GIF
    gif_dir = tmp_path / "assets"
    gif_dir.mkdir()
    (gif_dir / "A.gif").write_bytes(b"GIF89a")

    mgr = TutorialManager()
    # Override gifs_dir for test
    mgr.gifs_dir = gif_dir
    available = mgr._scan_available_gifs()
    assert "A" in available


def test_hint_messages():
    mgr = TutorialManager()
    msg1 = mgr.get_hint_message("A", 1)
    assert "'A'" in msg1
