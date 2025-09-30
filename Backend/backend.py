# aibetween_backend.py
import uuid
import json
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from Backend.backend_config.media_generator import generate_video_from_frame

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------

# ---------- CONFIG ----------
ROOT = Path(__file__).parent
UPLOAD_DIR = ROOT / "uploads"
UI_DIR = ROOT / "aibetween"
TIMELINE_DIR = UPLOAD_DIR / "timelines"
UPLOAD_DIR.mkdir(exist_ok=True)
UI_DIR.mkdir(exist_ok=True)
TIMELINE_DIR.mkdir(exist_ok=True)
# ----------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded media (so frontend can fetch /uploads/...) and UI folder
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/aibetween", StaticFiles(directory=str(UI_DIR)), name="aibetween")


# -------------------------
# Timeline / Clip helpers
# -------------------------
def timeline_path(timeline_id: str) -> Path:
    return TIMELINE_DIR / f"{timeline_id}.json"


def save_timeline(timeline_id: str, timeline: Dict):
    p = timeline_path(timeline_id)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2)
    except Exception as e:
        logger.exception(f"Error saving timeline {timeline_id}: {e}")


def load_timeline(timeline_id: str) -> Dict:
    p = timeline_path(timeline_id)
    if not p.exists():
        logger.error(f"Timeline file not found: {p}")
        raise FileNotFoundError("Timeline not found")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception(f"Error loading timeline {timeline_id}: {e}")
        raise


def clip_duration(clip: Dict) -> float:
    try:
        return float(clip["end"]) - float(clip["start"])
    except (KeyError, ValueError) as e:
        logger.exception(f"Error calculating clip duration for clip {clip}: {e}")
        raise


def total_timeline_duration(timeline: Dict) -> float:
    try:
        return sum(clip_duration(c) for c in timeline["timeline"])
    except Exception as e:
        logger.exception(f"Error calculating total timeline duration: {e}")
        raise


def make_clip_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# find clip by clip_id -> returns (index, clip)
def find_clip_index(timeline: Dict, clip_id: str) -> Tuple[Optional[int], Optional[Dict]]:
    for i, c in enumerate(timeline["timeline"]):
        if c["clip_id"] == clip_id:
            return i, c
    logger.error(f"Clip {clip_id} not found in timeline")
    return None, None


# -------------------------
# Frame extraction helper
# -------------------------
def extract_frame_to_png(video_path: str, time_seconds: float, out_png: Path) -> Path:
    """
    Extract a single frame at time_seconds (float) and save to out_png using ffmpeg.
    """
    # ffmpeg -y -ss TIME -i INPUT -frames:v 1 -q:v 2 OUT.png
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{time_seconds}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_png)
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            error_msg = f"ffmpeg frame extraction failed: {proc.stderr.decode()}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        return out_png
    except Exception as e:
        logger.exception(f"Error during frame extraction from {video_path} at {time_seconds}: {e}")
        raise


# -------------------------
# Split / Insert logic
# -------------------------
def split_clip_and_insert(timeline: Dict, clip_index: int, split_local_time: float,
                          new_clip_url: str, new_clip_duration: float) -> Dict:
    """
    Given a timeline, split the clip at clip_index at split_local_time (local time inside that clip file)
    and insert a new AI clip (new_clip_url with duration new_clip_duration) between before and after.
    clip object structure:
    {
      "clip_id": str,
      "type": "original"|"ai",
      "url": "/uploads/xxx.mp4",   # path string
      "start": float,   # local start in file
      "end": float
    }
    Behavior:
    - Create before (if length > 0): same url, start -> split_local_time
    - Create new AI clip: clip_id ai_x
    - Create after (if length > 0): same url, split_local_time -> end
    - Replace timeline[clip_index] with new sequence keeping order.
    """
    try:
        old = timeline["timeline"][clip_index]
        old_start = float(old["start"])
        old_end = float(old["end"])

        if not (old_start <= split_local_time <= old_end):
            error_msg = "split_local_time is outside clip range"
            logger.error(error_msg)
            raise ValueError(error_msg)

        before_len = split_local_time - old_start
        after_len = old_end - split_local_time

        new_items = []

        prefix = "original" if old["type"] == "original" else "ai"

        if before_len > 1e-6:
            before_id = make_clip_id(prefix)
            before_clip = {
                "clip_id": before_id,
                "type": old["type"],
                "url": old["url"],
                "start": old_start,
                "end": split_local_time
            }
            new_items.append(before_clip)

        # new AI clip
        ai_id = make_clip_id("ai")
        ai_clip = {
            "clip_id": ai_id,
            "type": "ai",
            "url": new_clip_url,
            "start": 0.0,
            "end": float(new_clip_duration)
        }
        new_items.append(ai_clip)

        if after_len > 1e-6:
            after_id = make_clip_id(prefix)
            after_clip = {
                "clip_id": after_id,
                "type": old["type"],
                "url": old["url"],
                "start": split_local_time,
                "end": old_end
            }
            new_items.append(after_clip)

        # Replace in place
        timeline["timeline"][clip_index:clip_index + 1] = new_items
        return timeline
    except Exception as e:
        logger.exception(
            f"Error in split_clip_and_insert for clip_index {clip_index}, split_time {split_local_time}: {e}")
        raise


# A helper that splits an existing clip but when new insertion happens inside an ai clip
# and you want before/after to reference same ai file (we already do above).
# Caller should supply new_clip_url and duration.

# -------------------------
# API Endpoints
# -------------------------
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video and create an initial timeline with a single original clip spanning the whole file.
    Returns timeline_id and timeline JSON.
    """
    try:
        out_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
        with open(out_path, "wb") as f:
            f.write(await file.read())

        # Probe duration with ffprobe
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(out_path)
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            error_msg = "ffprobe failed"
            logger.error(f"{error_msg}: {proc.stderr.decode()}")
            raise HTTPException(status_code=500, detail=error_msg)
        duration = float(proc.stdout.decode().strip())

        timeline_id = f"timeline_{uuid.uuid4().hex[:8]}"
        timeline = {
            "timeline_id": timeline_id,
            "source_file": str(out_path.name),
            "timeline": [
                {
                    "clip_id": make_clip_id("original"),
                    "type": "original",
                    "url": f"/uploads/{out_path.name}",
                    "start": 0.0,
                    "end": duration
                }
            ]
        }

        save_timeline(timeline_id, timeline)
        return JSONResponse({"timeline_id": timeline_id, "timeline": timeline})
    except Exception as e:
        logger.exception(f"Error in upload_video for file {file.filename}: {e}")
        raise


@app.get("/timeline/{timeline_id}")
async def get_timeline(timeline_id: str):
    try:
        timeline = load_timeline(timeline_id)
    except FileNotFoundError:
        logger.error(f"Timeline not found: {timeline_id}")
        raise HTTPException(status_code=404, detail="Timeline not found")
    except Exception as e:
        logger.exception(f"Error getting timeline {timeline_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    return JSONResponse(timeline)


@app.post("/generate-clip")
async def generate_clip(timeline_id: str = Form(...), clip_id: str = Form(...),
                        local_timestamp: float = Form(...), prompt: str = Form(...)):
    """
    Entry point from frontend when user stops on a clip and requests AI generation.
    local_timestamp is seconds into the clip identified by clip_id.
    """
    try:
        timeline = load_timeline(timeline_id)
    except FileNotFoundError:
        logger.error(f"Timeline not found: {timeline_id}")
        raise HTTPException(status_code=404, detail="Timeline not found")
    except Exception as e:
        logger.exception(f"Error loading timeline {timeline_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    idx, clip = find_clip_index(timeline, clip_id)
    if idx is None:
        logger.error(f"Clip not found in timeline {timeline_id}: {clip_id}")
        raise HTTPException(status_code=404, detail="Clip not found in timeline")

    # compute absolute time inside source file for frame extraction:
    # For an original or ai clip, clip.start is the local start in the file.
    # So the extraction time inside the file is clip.start + local_timestamp.
    try:
        clip_start = float(clip["start"])
        clip_end = float(clip["end"])
    except (KeyError, ValueError) as e:
        logger.exception(f"Invalid clip data for {clip_id}: {e}")
        raise HTTPException(status_code=500, detail="Invalid clip data")

    if not (0.0 <= local_timestamp <= (clip_end - clip_start) + 1e-6):
        logger.error(f"local_timestamp outside clip duration: {local_timestamp} for clip {clip_id}")
        raise HTTPException(status_code=400, detail="local_timestamp outside clip duration")

    extract_time_in_source = clip_start + local_timestamp
    source_url = clip["url"]
    # source_url may be relative (e.g., "/uploads/abc.mp4")
    source_path = ROOT / source_url.lstrip("/")

    # 1) extract frame
    tmp_png = Path(tempfile.gettempdir()) / f"frame_{uuid.uuid4().hex}.png"
    try:
        extract_frame_to_png(str(source_path), extract_time_in_source, tmp_png)
    except Exception as e:
        logger.exception(f"Frame extraction error for {source_path} at {extract_time_in_source}: {e}")
        raise HTTPException(status_code=500, detail=f"frame extraction error: {e}")

    # 2) generate AI clip (calls provided function)
    try:
        ai_out_path = generate_video_from_frame(str(tmp_png), prompt)
    except Exception as e:
        logger.exception(f"AI generation error: {e}")
        raise HTTPException(status_code=500, detail=f"AI generation error: {e}")
    finally:
        if tmp_png.exists():
            try:
                tmp_png.unlink()
            except Exception as e:
                logger.exception(f"Error cleaning up temp PNG {tmp_png}: {e}")

    # ai_out_path should be absolute; move/copy into uploads dir if not there
    ai_out = Path(ai_out_path)
    if ai_out.parent != UPLOAD_DIR:
        dest = UPLOAD_DIR / ai_out.name
        try:
            shutil.copy(ai_out, dest)
            ai_out = dest
        except Exception as e:
            logger.exception(f"Error copying AI output to uploads: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing AI output: {e}")

    # determine ai duration using ffprobe
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        str(ai_out)
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        error_msg = "ffprobe failed for generated clip"
        logger.error(f"{error_msg}: {proc.stderr.decode()}")
        raise HTTPException(status_code=500, detail=error_msg)
    ai_duration = float(proc.stdout.decode().strip())

    # 3) split the clip at the provided local timestamp and insert the new ai clip
    #    Note: split_local_time is in the clip's local coordinate system, i.e., clip.start..clip.end
    split_local_time = clip_start + local_timestamp  # local time inside source file
    # BUT our split function expects a split time in the clip's local coordinates (start..end)
    # So it's correct to use split_local_time directly because the clip uses local times already.
    # Example: clip.start = 0, clip.end = 10.5 -> split_local_time = 0 + 10.5 = 10.5

    # However, earlier functions treat clip.start/end as the times within the file (not global),
    # and we split using those same values — so all consistent.

    # call split/insert
    # Note: we store URLs as "/uploads/filename"
    new_clip_rel_url = f"/uploads/{ai_out.name}"
    try:
        updated = split_clip_and_insert(timeline, idx, split_local_time, new_clip_rel_url, ai_duration)
    except Exception as e:
        logger.exception(f"Split/insert error for timeline {timeline_id}, clip {clip_id}: {e}")
        raise HTTPException(status_code=500, detail=f"split/insert error: {e}")

    # save updated timeline
    save_timeline(timeline_id, timeline)

    return JSONResponse({"timeline_id": timeline_id, "timeline": timeline})


@app.post("/export/{timeline_id}")
async def export_timeline(timeline_id: str):
    """
    Create a final exported MP4 that stitches the timeline clips in order.
    This will trim ranges from clip urls and concatenate them.
    """
    try:
        timeline = load_timeline(timeline_id)
    except FileNotFoundError:
        logger.error(f"Timeline not found for export: {timeline_id}")
        raise HTTPException(status_code=404, detail="Timeline not found")
    except Exception as e:
        logger.exception(f"Error loading timeline for export {timeline_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    parts: List[Path] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="aibetween_export_"))

    try:
        for i, seg in enumerate(timeline["timeline"]):
            try:
                start = float(seg["start"])
                end = float(seg["end"])
                dur = end - start
                if dur <= 1e-6:
                    continue

                src = ROOT / seg["url"].lstrip("/")
                out_part = tmpdir / f"part_{i}_{uuid.uuid4().hex}.mp4"

                # ffmpeg -y -i src -ss start -t dur -c copy out_part
                # Using -ss after -i for accuracy
                cmd = [
                    "ffmpeg", "-y", "-i", str(src),
                    "-ss", str(start),
                    "-t", str(dur),
                    "-c", "copy",
                    str(out_part)
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if proc.returncode != 0:
                    # If copy fails (container mismatch), re-encode as fallback:
                    cmd2 = [
                        "ffmpeg", "-y", "-i", str(src),
                        "-ss", str(start),
                        "-t", str(dur),
                        "-c:v", "libx264", "-c:a", "aac",
                        str(out_part)
                    ]
                    proc2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if proc2.returncode != 0:
                        error_msg = f"ffmpeg trim failed: {proc.stderr.decode()} / {proc2.stderr.decode()}"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                parts.append(out_part)
            except Exception as e:
                logger.exception(f"Error processing segment {i} in export for {timeline_id}: {e}")
                raise

        # create concat file
        concat_list = tmpdir / "concat.txt"
        try:
            with open(concat_list, "w", encoding="utf-8") as f:
                for p in parts:
                    f.write(f"file '{p.as_posix()}'\n")
        except Exception as e:
            logger.exception(f"Error creating concat list for export {timeline_id}: {e}")
            raise

        out_final = UPLOAD_DIR / f"export_{timeline_id}_{uuid.uuid4().hex}.mp4"
        cmd_concat = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(out_final)
        ]
        proc = subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            # fallback re-encode concat
            cmd_concat2 = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c:v", "libx264", "-c:a", "aac",
                str(out_final)
            ]
            proc2 = subprocess.run(cmd_concat2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc2.returncode != 0:
                error_msg = f"ffmpeg concat failed: {proc.stderr.decode()} / {proc2.stderr.decode()}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        return JSONResponse({"export_url": f"/uploads/{out_final.name}"})
    except Exception as e:
        logger.exception(f"Error during export for {timeline_id}: {e}")
        raise HTTPException(status_code=500, detail="Export failed")
    finally:
        # optionally cleanup tmpdir - keep for debugging in dev
        # shutil.rmtree(tmpdir)
        pass


# small helper to compute overall timeline total duration (frontend uses too)
@app.get("/timeline-duration/{timeline_id}")
async def timeline_duration(timeline_id: str):
    try:
        timeline = load_timeline(timeline_id)
    except FileNotFoundError:
        logger.error(f"Timeline not found for duration: {timeline_id}")
        raise HTTPException(status_code=404, detail="Timeline not found")
    except Exception as e:
        logger.exception(f"Error getting timeline duration for {timeline_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    return JSONResponse({"duration": total_timeline_duration(timeline)})


@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse(url="/aibetween/index.html")