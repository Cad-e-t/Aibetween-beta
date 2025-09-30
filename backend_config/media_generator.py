import os
import uuid
import shutil
from pathlib import Path
from google.genai import types
import time
import json, logging
from google import genai
from pyarrow import duration

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------

google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ---------- CONFIG ----------
ROOT = Path(__file__).parent
UPLOAD_DIR = ROOT / "uploads"
UI_DIR = ROOT / "aibetween"
UPLOAD_DIR.mkdir(exist_ok=True)
UI_DIR.mkdir(exist_ok=True)  # ensure exists (you'll put index.html here)

TEST_MODE = False     # toggle to False to call Veo
TEST_CLIP = UPLOAD_DIR / "test_clip.mp4"  # must exist if TEST_MODE=True
# ----------------------------



def generate_video_from_frame(image_path: str, prompt: str, aspect_ratio: str = "16:9") -> str:
    """
    Returns local path to generated AI video.
    Uses TEST_MODE copy if enabled; otherwise calls Veo and saves output to uploads/.
    """
    logger = logging.getLogger(__name__)
    try:
        # TEST MODE: reuse a local clip so you don't spend credits
        if TEST_MODE:
            test_clip = UPLOAD_DIR / "test_clip.mp4"
            if not test_clip.exists():
                error_msg = "TEST_MODE is on but uploads/test_clip.mp4 is missing."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            out = UPLOAD_DIR / f"ai_{uuid.uuid4().hex}.mp4"
            shutil.copy(test_clip, out)
            return str(out)

        # --- REAL Veo path ---
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        image_object = types.Image(
            image_bytes=image_bytes,
            mime_type="image/png"
        )

        operation = google_client.models.generate_videos(
            model="veo-3.0-fast-generate-001",
            prompt=prompt,
            image=image_object,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                person_generation="allow_adult"
            )
        )

        while not operation.done:
            time.sleep(5)
            operation = google_client.operations.get(operation)

        # Ensure we got a valid response
        if not operation.response:
            error_msg = "Video generation failed: no response from API."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        generated_videos = getattr(operation.response, "generated_videos", None)
        if not generated_videos or not generated_videos[0].video:
            error_msg = "Video generation failed: no video returned."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        file_bytes = google_client.files.download(file=generated_videos[0].video)
        out_path = UPLOAD_DIR / f"ai_{uuid.uuid4().hex}.mp4"
        with open(out_path, "wb") as out_f:
            out_f.write(file_bytes)

        return str(out_path)
    except Exception as e:
        logger.exception(f"Error in generate_video_from_frame for image {image_path} and prompt '{prompt}': {e}")
        raise

