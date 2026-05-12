from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.backend.core import frontend_build


class FrontendBuildTests(unittest.TestCase):
    def test_skips_build_when_export_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frontend_dir = root / "frontend"
            out_dir = frontend_dir / "out"
            out_dir.mkdir(parents=True)
            (out_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            (frontend_dir / "package.json").write_text("{}", encoding="utf-8")

            with patch.object(frontend_build, "PROJECT_ROOT", root), patch.object(frontend_build, "FRONTEND_DIR", frontend_dir), patch.object(frontend_build, "FRONTEND_OUT_DIR", out_dir), patch.object(frontend_build, "FRONTEND_INDEX_FILE", out_dir / "index.html"), patch("src.backend.core.frontend_build.subprocess.run") as run_mock:
                self.assertEqual(frontend_build.ensure_frontend_export(), out_dir)
                run_mock.assert_not_called()

    def test_builds_when_export_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frontend_dir = root / "frontend"
            out_dir = frontend_dir / "out"
            frontend_dir.mkdir(parents=True)
            (frontend_dir / "package.json").write_text("{}", encoding="utf-8")

            def fake_run(*args, **kwargs):
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "index.html").write_text("<html></html>", encoding="utf-8")

                class Result:
                    returncode = 0

                return Result()

            with patch.object(frontend_build, "PROJECT_ROOT", root), patch.object(frontend_build, "FRONTEND_DIR", frontend_dir), patch.object(frontend_build, "FRONTEND_OUT_DIR", out_dir), patch.object(frontend_build, "FRONTEND_INDEX_FILE", out_dir / "index.html"), patch("src.backend.core.frontend_build.shutil.which", return_value="npm"), patch("src.backend.core.frontend_build.subprocess.run", side_effect=fake_run) as run_mock:
                self.assertEqual(frontend_build.ensure_frontend_export(), out_dir)
                run_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
