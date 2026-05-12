"""
Start script with proper error handling and logging.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    print("=" * 60)
    print("Starting Sign Language Learning App")
    print("=" * 60)
    
    try:
        import uvicorn
        from src.backend.models.downloader import check_and_download_models
        from src.backend.core.frontend_build import ensure_frontend_export
        
        print("\n Imports successful")
        print(f" Project root: {project_root}")
        
        # Check and download models before starting server
        check_and_download_models()

        # Build the frontend export so FastAPI can serve the app at /
        ensure_frontend_export()

        from src.backend.api.routes import app
        
        print("\nStarting server on http://localhost:8000")
        print("Press Ctrl+C to stop\n")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"\n Import error: {e}")
        print("\nPlease install dependencies:")
        print("  uv sync")
        sys.exit(1)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
