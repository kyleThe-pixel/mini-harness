# Photo Tagger v1

Local photo search for your own drive. Point it at a folder, it scans every
photo and indexes:

- **Semantic tags** — scene, objects, people, colors, mood, setting
  (via your llama-server vision model, or local CLIP as offline fallback)
- **Text in images** — OCR (EasyOCR), so a word on a sign/photo is searchable
- **EXIF** — date taken, camera, GPS coordinates when present
- **Filenames** — always searchable

Everything is stored in `data/index.db` (SQLite + full-text search) next to
the app. Thumbnails live in `data/thumbs/`. Your photos are never modified,
moved, or uploaded anywhere.

## Setup (Windows)

1. Put this folder anywhere, e.g. `C:\PhotoTagger`
2. Double-click **`setup_tagger_v1.bat`** (one time — adds packages to your
   Forge Neo portable Python, which already has torch/CUDA + gradio)
3. Double-click **`start_tagger_v1.bat`** → opens http://127.0.0.1:7862

## Use

1. **Library tab** — paste your photo folder, Save, then Scan.
   Re-scans skip unchanged files automatically.
2. **Search tab** — type anything: `beach`, `dog`, `women`, a word from a
   sign in a photo, a filename. Click a result for full details.
3. **Settings tab** — vision backend:
   - `api` (default): your llama-server router at http://127.0.0.1:8080/v1.
     Load a vision-capable model there first (e.g. Qwen2.5-VL).
     Use **Test API connection** to verify.
   - `clip`: offline fallback, no server needed, dumber tags.

## Notes

- First scan downloads EasyOCR models (~100MB) once.
- Big libraries take a while (a few seconds per photo on the API backend).
  Let the first scan run; later scans are incremental.
- HEIC (iPhone) photos work via pillow-heif.
- If you ever reinstall Forge's Python, re-run `setup_tagger_v1.bat`.
