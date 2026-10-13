"""
Photo Tagger - local scan engine.
Walks a photo library, extracts EXIF + OCR text + semantic tags,
and stores everything in a SQLite DB with full-text search.

Vision backends:
  - "api"  : any OpenAI-compatible vision endpoint
             (default: James's llama-server router at http://127.0.0.1:8080/v1)
  - "clip" : local CLIP zero-shot tagging, no server needed (fallback)

Everything stays on this machine. No cloud, no uploads.
"""
import base64
import hashlib
import io
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ExifTags

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CONFIG_PATH = APP_DIR / "config.json"
DB_PATH = DATA_DIR / "index.db"
THUMB_DIR = DATA_DIR / "thumbs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif"}

DEFAULT_CONFIG = {
    "library_dir": "",
    "backend": "api",                       # "api" | "clip"
    "api_base": "http://127.0.0.1:8080/v1", # llama-server router
    "api_model": "",                        # empty = server default / first loaded
    "api_timeout": 300,
    "ocr_enabled": True,
    "max_image_px": 1568,                   # downscale before sending to VLM/OCR
    "thumb_px": 320,
}

# ---------------------------------------------------------------- config ---

def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# ------------------------------------------------------------------- db ---

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id           INTEGER PRIMARY KEY,
    path         TEXT UNIQUE,
    filename     TEXT,
    width        INTEGER,
    height       INTEGER,
    date_taken   TEXT,
    camera       TEXT,
    gps_lat      REAL,
    gps_lon      REAL,
    tags         TEXT,          -- comma-separated semantic tags
    ocr_text     TEXT,          -- text found in the image
    people_count INTEGER,
    setting      TEXT,          -- indoor / outdoor / unknown
    backend      TEXT,
    file_mtime   REAL,
    file_size    INTEGER,
    scanned_at   TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS photos_fts USING fts5(
    filename, tags, ocr_text,
    content='photos', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS photos_ai AFTER INSERT ON photos BEGIN
    INSERT INTO photos_fts(rowid, filename, tags, ocr_text)
    VALUES (new.id, new.filename, new.tags, new.ocr_text);
END;
CREATE TRIGGER IF NOT EXISTS photos_ad AFTER DELETE ON photos BEGIN
    INSERT INTO photos_fts(photos_fts, rowid, filename, tags, ocr_text)
    VALUES ('delete', old.id, old.filename, old.tags, old.ocr_text);
END;
CREATE TRIGGER IF NOT EXISTS photos_au AFTER UPDATE ON photos BEGIN
    INSERT INTO photos_fts(photos_fts, rowid, filename, tags, ocr_text)
    VALUES ('delete', old.id, old.filename, old.tags, old.ocr_text);
    INSERT INTO photos_fts(rowid, filename, tags, ocr_text)
    VALUES (new.id, new.filename, new.tags, new.ocr_text);
END;
"""


def get_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def fts_safe_query(q):
    """Turn free text into a safe FTS5 query (AND of quoted terms)."""
    terms = [t.strip().lower() for t in q.replace(",", " ").split() if t.strip()]
    if not terms:
        return None
    return " AND ".join('"%s"' % t.replace('"', "") for t in terms)


def search_db(query, limit=60):
    conn = get_db()
    conn.row_factory = sqlite3.Row
    if not query or not query.strip():
        rows = conn.execute(
            "SELECT * FROM photos ORDER BY scanned_at DESC LIMIT ?", (limit,)
        ).fetchall()
    else:
        match = fts_safe_query(query)
        if not match:
            return []
        rows = conn.execute(
            """SELECT p.* FROM photos_fts f
               JOIN photos p ON p.id = f.rowid
               WHERE photos_fts MATCH ? LIMIT ?""",
            (match, limit),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_stats():
    conn = get_db()
    n = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    last = conn.execute("SELECT MAX(scanned_at) FROM photos").fetchone()[0]
    tagged = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE tags IS NOT NULL AND tags != ''"
    ).fetchone()[0]
    conn.close()
    return {"total": n, "tagged": tagged, "last_scan": last}


def thumb_path(photo_id):
    return THUMB_DIR / f"{photo_id}.jpg"


# ----------------------------------------------------------------- exif ---

def _dms_to_deg(dms, ref):
    try:
        d = float(dms[0]) + float(dms[1]) / 60 + float(dms[2]) / 3600
        if ref in ("S", "W"):
            d = -d
        return d
    except Exception:
        return None


def read_exif(img):
    out = {"date_taken": None, "camera": None, "gps_lat": None, "gps_lon": None}
    try:
        raw = img.getexif()
        if not raw:
            return out
        exif = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
        dt = exif.get("DateTimeOriginal") or exif.get("DateTime")
        if dt:
            try:
                out["date_taken"] = datetime.strptime(str(dt), "%Y:%m:%d %H:%M:%S").isoformat()
            except Exception:
                out["date_taken"] = str(dt)
        make = exif.get("Make", "")
        model = exif.get("Model", "")
        cam = f"{make} {model}".strip()
        out["camera"] = cam or None
        gps = exif.get("GPSInfo")
        if gps:
            lat = _dms_to_deg(gps.get(2), gps.get(1))
            lon = _dms_to_deg(gps.get(4), gps.get(3))
            out["gps_lat"], out["gps_lon"] = lat, lon
    except Exception:
        pass
    return out


# ------------------------------------------------------------------ ocr ---

_ocr_reader = None


def ocr_image(pil_img):
    """Return text found in the image (EasyOCR). None if unavailable/disabled."""
    global _ocr_reader
    try:
        import easyocr
    except ImportError:
        return None
    try:
        if _ocr_reader is None:
            _ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    except Exception:
        try:
            _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception:
            return None
    try:
        import numpy as np
        res = _ocr_reader.readtext(np.array(pil_img))
        texts = [t[1].strip() for t in res if t[1] and t[1].strip()]
        # de-dup while keeping order
        seen, uniq = set(), []
        for t in texts:
            k = t.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(t)
        return " | ".join(uniq) if uniq else ""
    except Exception:
        return None


# ---------------------------------------------------------- vision: api ---

VISION_PROMPT = """Analyze this photo and reply with ONLY a JSON object, no other text:
{
  "tags": ["tag1", "tag2", ...],
  "people_count": 0,
  "setting": "indoor" | "outdoor" | "unknown",
  "text_in_image": "any visible text, or empty string"
}
Rules for tags (aim for 10-20, lowercase, single words or short phrases):
- scene type: beach, city street, forest, kitchen, concert, wedding, etc.
- main objects: dog, car, bicycle, cake, laptop, etc.
- people: "1 man", "2 women", "group of friends", "baby", "crowd" (be specific)
- animals, food, vehicles, buildings, nature elements
- colors that dominate, time of day (morning, sunset, night), weather
- mood/activity: hiking, cooking, party, sports, relaxing
- photo type: selfie, screenshot, portrait, landscape, close-up, document
No sentences, no commentary, just the JSON."""


def tags_via_api(pil_img, cfg):
    import requests

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}
        ],
        "max_tokens": 600,
        "temperature": 0.1,
    }
    if cfg.get("api_model"):
        body["model"] = cfg["api_model"]
    url = cfg["api_base"].rstrip("/") + "/chat/completions"
    r = requests.post(url, json=body, timeout=cfg.get("api_timeout", 300))
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return parse_vision_json(content)


def parse_vision_json(content):
    """Pull the JSON object out of a model reply, defensively."""
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        return {"tags": [], "people_count": 0, "setting": "unknown",
                "text_in_image": "", "_raw": content}
    try:
        data = json.loads(content[start:end + 1])
    except Exception:
        return {"tags": [], "people_count": 0, "setting": "unknown",
                "text_in_image": "", "_raw": content}
    tags = data.get("tags") or []
    tags = [str(t).strip().lower() for t in tags if str(t).strip()]
    try:
        pc = int(data.get("people_count") or 0)
    except Exception:
        pc = 0
    setting = str(data.get("setting") or "unknown").lower()
    if setting not in ("indoor", "outdoor"):
        setting = "unknown"
    return {"tags": tags, "people_count": pc, "setting": setting,
            "text_in_image": str(data.get("text_in_image") or "").strip()}


# --------------------------------------------------------- vision: clip ---

CLIP_VOCAB = (
    # scenes
    "beach ocean sea lake river mountain forest desert city street park garden "
    "kitchen bedroom living room bathroom office restaurant cafe bar concert stadium "
    "wedding party festival market gym pool snow ski airport highway bridge "
    # objects
    "car truck bicycle motorcycle dog cat bird horse food pizza cake coffee laptop "
    "phone book guitar piano flowers tree boat airplane train ship baby stroller "
    # people
    "man woman child baby group of people crowd selfie portrait family friends "
    # setting / light
    "indoor outdoor daytime nighttime sunset sunrise morning evening "
    # photo type
    "screenshot document close-up landscape blurry black and white colorful"
)

_clip = None


def tags_via_clip(pil_img):
    """Zero-shot CLIP tagging against a fixed vocabulary. Offline fallback."""
    global _clip
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        return {"tags": [], "people_count": 0, "setting": "unknown",
                "text_in_image": ""}
    if _clip is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip = (model, proc)
    model, proc = _clip
    vocab = CLIP_VOCAB.split()
    inputs = proc(text=vocab, images=pil_img, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    probs = out.logits_per_image.softmax(dim=1)[0]
    top = sorted(zip(vocab, probs.tolist()), key=lambda x: -x[1])[:14]
    tags = [t for t, p in top if p > 0.02]
    people = sum(1 for t in tags if t in
                 ("man", "woman", "child", "baby", "group of people", "crowd",
                  "family", "friends", "selfie", "portrait"))
    setting = "unknown"
    if "indoor" in tags:
        setting = "indoor"
    elif "outdoor" in tags:
        setting = "outdoor"
    return {"tags": tags, "people_count": people, "setting": setting,
            "text_in_image": ""}


# ----------------------------------------------------------------- scan ---

def iter_images(library_dir):
    lib = Path(library_dir)
    if not lib.is_dir():
        return
    for p in sorted(lib.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            # skip our own data dir if someone points at the app folder
            if ".phototagger" in p.parts or p.suffix.lower() in (".db",):
                continue
            yield p


def open_image(path):
    suffix = Path(path).suffix.lower()
    if suffix in (".heic", ".heif"):
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            return None
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


def downscale(img, max_px):
    w, h = img.size
    m = max(w, h)
    if m > max_px:
        img = img.resize((int(w * max_px / m), int(h * max_px / m)), Image.LANCZOS)
    return img


def make_thumb(img, photo_id, thumb_px=320):
    t = img.copy()
    t.thumbnail((thumb_px, thumb_px), Image.LANCZOS)
    tp = thumb_path(photo_id)
    t.save(tp, "JPEG", quality=80)
    return tp


def scan_library(library_dir, cfg=None, on_progress=None, rescan=False):
    """Full or incremental scan. Yields (done, total, current_file)."""
    cfg = cfg or load_config()
    lib = str(Path(library_dir))
    cfg["library_dir"] = lib
    save_config(cfg)

    files = list(iter_images(lib))
    total = len(files)
    conn = get_db()
    done = 0
    for f in files:
        done += 1
        if on_progress:
            on_progress(done, total, f.name)
        st = f.stat()
        row = conn.execute(
            "SELECT id, file_mtime, file_size FROM photos WHERE path = ?",
            (str(f),)).fetchone()
        if row and not rescan and row[1] == st.st_mtime and row[2] == st.st_size:
            continue  # unchanged, skip
        photo_id = scan_one(conn, f, st, cfg, existing_id=row[0] if row else None)
        if (done % 25) == 0:
            conn.commit()
    conn.commit()
    # prune entries whose files vanished
    conn.execute(
        "DELETE FROM photos WHERE path NOT IN (%s)" % ",".join("?" * total),
        [str(f) for f in files]) if total else conn.execute("DELETE FROM photos")
    conn.commit()
    conn.close()
    return {"scanned": done, "total": total}


def scan_one(conn, path, st, cfg, existing_id=None):
    img = open_image(path)
    if img is None:
        return None
    w, h = img.size
    exif = read_exif(img)
    small = downscale(img, cfg.get("max_image_px", 1568))

    # OCR
    ocr_text = ""
    if cfg.get("ocr_enabled", True):
        ocr_text = ocr_image(small) or ""

    # semantic tags
    backend = cfg.get("backend", "api")
    result = {"tags": [], "people_count": 0, "setting": "unknown",
              "text_in_image": ""}
    try:
        if backend == "clip":
            result = tags_via_clip(small)
        else:
            result = tags_via_api(small, cfg)
    except Exception as e:
        result["_error"] = str(e)[:200]

    # merge VLM-transcribed text into OCR field when EasyOCR found nothing
    if not ocr_text and result.get("text_in_image"):
        ocr_text = result["text_in_image"]

    tags = ", ".join(result.get("tags", []))
    now = datetime.now().isoformat(timespec="seconds")
    if existing_id:
        conn.execute(
            """UPDATE photos SET filename=?, width=?, height=?, date_taken=?,
               camera=?, gps_lat=?, gps_lon=?, tags=?, ocr_text=?, people_count=?,
               setting=?, backend=?, file_mtime=?, file_size=?, scanned_at=?
               WHERE id=?""",
            (path.name, w, h, exif["date_taken"], exif["camera"],
             exif["gps_lat"], exif["gps_lon"], tags, ocr_text,
             result.get("people_count", 0), result.get("setting", "unknown"),
             backend, st.st_mtime, st.st_size, now, existing_id))
        photo_id = existing_id
    else:
        cur = conn.execute(
            """INSERT INTO photos (path, filename, width, height, date_taken,
               camera, gps_lat, gps_lon, tags, ocr_text, people_count, setting,
               backend, file_mtime, file_size, scanned_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (str(path), path.name, w, h, exif["date_taken"], exif["camera"],
             exif["gps_lat"], exif["gps_lon"], tags, ocr_text,
             result.get("people_count", 0), result.get("setting", "unknown"),
             backend, st.st_mtime, st.st_size, now))
        photo_id = cur.lastrowid
    try:
        make_thumb(img, photo_id, cfg.get("thumb_px", 320))
    except Exception:
        pass
    return photo_id


def get_photo(photo_id):
    conn = get_db()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM photos WHERE id=?", (photo_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
