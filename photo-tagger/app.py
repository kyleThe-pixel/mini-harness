"""
Photo Tagger UI - local photo search.
Search bar over everything the scanner indexed: semantic tags,
OCR text ("words on the image"), filenames, dates.
"""
import gradio as gr

import tagger
from tagger import load_config, save_config, search_db, db_stats, get_photo, thumb_path


def fmt_photo_info(p):
    if not p:
        return ""
    lines = [f"**{p['filename']}**", ""]
    lines.append(f"Size: {p['width']}x{p['height']}")
    if p.get("date_taken"):
        lines.append(f"Taken: {p['date_taken']}")
    if p.get("camera"):
        lines.append(f"Camera: {p['camera']}")
    if p.get("gps_lat") is not None:
        lines.append(f"GPS: {p['gps_lat']:.4f}, {p['gps_lon']:.4f}")
    if p.get("people_count"):
        lines.append(f"People: {p['people_count']}")
    if p.get("setting") and p["setting"] != "unknown":
        lines.append(f"Setting: {p['setting']}")
    lines.append("")
    lines.append(f"**Tags:** {p['tags'] or '(none)'}")
    lines.append("")
    lines.append(f"**Text in image:** {p['ocr_text'] or '(none)'}")
    lines.append("")
    lines.append(f"Path: `{p['path']}`")
    return "\n".join(lines)


def do_search(query):
    results = search_db(query, limit=80)
    tagger_state = results
    gallery = []
    for r in results:
        tp = thumb_path(r["id"])
        src = str(tp) if tp.exists() else r["path"]
        gallery.append((src, r["filename"]))
    status = f"{len(results)} result(s)" + (f" for '{query}'" if query else "")
    return gallery, tagger_state, status


def show_detail(evt: gr.SelectData, state):
    if not state or evt.index is None or evt.index >= len(state):
        return None, ""
    p = get_photo(state[evt.index]["id"])
    if not p:
        return None, ""
    return p["path"], fmt_photo_info(p)


def do_scan(rescan, progress=gr.Progress(track_tqdm=True)):
    cfg = load_config()
    lib = cfg.get("library_dir", "").strip()
    if not lib:
        return "Set a library folder first (Library tab)."
    from pathlib import Path
    if not Path(lib).is_dir():
        return f"Folder not found: {lib}"

    def cb(done, total, name):
        progress((done, total), desc=f"{done}/{total}: {name}")

    try:
        out = tagger.scan_library(lib, cfg, on_progress=cb, rescan=rescan)
    except Exception as e:
        return f"Scan failed: {e}"
    s = db_stats()
    return (f"Done. {out['total']} files checked. "
            f"Library now holds {s['total']} photos ({s['tagged']} tagged).")


def save_library(library_dir):
    cfg = load_config()
    cfg["library_dir"] = library_dir.strip()
    save_config(cfg)
    return f"Library folder saved: {cfg['library_dir'] or '(none)'}"


def save_settings(backend, api_base, api_model, ocr_enabled, max_px):
    cfg = load_config()
    cfg["backend"] = backend
    cfg["api_base"] = api_base.strip().rstrip("/")
    cfg["api_model"] = api_model.strip()
    cfg["ocr_enabled"] = bool(ocr_enabled)
    try:
        cfg["max_image_px"] = max(512, int(max_px))
    except Exception:
        pass
    save_config(cfg)
    return "Settings saved."


def test_api(api_base):
    import requests
    try:
        r = requests.get(api_base.strip().rstrip("/") + "/models", timeout=15)
        r.raise_for_status()
        models = [m.get("id", "?") for m in r.json().get("data", [])]
        return "API reachable. Models: " + (", ".join(models[:8]) or "(none listed)")
    except Exception as e:
        return f"API not reachable: {e}"


def build_ui():
    cfg = load_config()
    with gr.Blocks(title="Photo Tagger") as demo:
        gr.Markdown("# Photo Tagger\nLocal photo search — tags, text in images, filenames. Nothing leaves this machine.")
        state = gr.State([])

        with gr.Tab("Search"):
            with gr.Row():
                q = gr.Textbox(label="Search", placeholder="beach  sunset  dog ... or a word that appears in the image",
                               scale=4)
                btn = gr.Button("Search", variant="primary", scale=1)
            status = gr.Markdown("")
            with gr.Row():
                gallery = gr.Gallery(label="Results", columns=4, height=480, scale=3)
                with gr.Column(scale=2):
                    detail_img = gr.Image(label="Photo", height=480)
                    detail_info = gr.Markdown("")
            btn.click(do_search, inputs=q, outputs=[gallery, state, status])
            q.submit(do_search, inputs=q, outputs=[gallery, state, status])
            gallery.select(show_detail, inputs=state, outputs=[detail_img, detail_info])

        with gr.Tab("Library"):
            gr.Markdown("Point the tagger at a photo folder, then scan. Re-scans skip unchanged files.")
            lib_dir = gr.Textbox(label="Photo library folder",
                                 value=cfg.get("library_dir", ""),
                                 placeholder=r"C:\Users\tiger\Pictures")
            with gr.Row():
                save_lib_btn = gr.Button("Save folder")
                scan_btn = gr.Button("Scan (new/changed)", variant="primary")
                rescan_btn = gr.Button("Rescan everything")
            lib_status = gr.Markdown("")
            stats_btn = gr.Button("Refresh stats")
            stats_md = gr.Markdown("")
            save_lib_btn.click(save_library, inputs=lib_dir, outputs=lib_status)
            scan_btn.click(do_scan, inputs=gr.State(False), outputs=lib_status)
            rescan_btn.click(do_scan, inputs=gr.State(True), outputs=lib_status)
            stats_btn.click(lambda: (
                f"**{db_stats()['total']}** photos indexed "
                f"({db_stats()['tagged']} with tags) · "
                f"last scan: {db_stats()['last_scan'] or 'never'}"), outputs=stats_md)

        with gr.Tab("Settings"):
            gr.Markdown("Vision backend used when scanning. `llama-server` = your local "
                        "llama.cpp router (OpenAI-compatible). `local CLIP` = offline fallback, "
                        "no server needed but dumber tags.")
            backend = gr.Dropdown(["api", "clip"], value=cfg.get("backend", "api"),
                                  label="Vision backend (api = llama-server, clip = local fallback)")
            api_base = gr.Textbox(label="API base URL", value=cfg.get("api_base", ""))
            api_model = gr.Textbox(label="Model name (empty = server default)",
                                   value=cfg.get("api_model", ""))
            ocr_enabled = gr.Checkbox(label="OCR text detection (EasyOCR)",
                                      value=cfg.get("ocr_enabled", True))
            max_px = gr.Number(label="Max image size sent to vision (px)",
                               value=cfg.get("max_image_px", 1568), precision=0)
            with gr.Row():
                save_btn = gr.Button("Save settings", variant="primary")
                test_btn = gr.Button("Test API connection")
            set_status = gr.Markdown("")
            save_btn.click(save_settings,
                           inputs=[backend, api_base, api_model, ocr_enabled, max_px],
                           outputs=set_status)
            test_btn.click(test_api, inputs=api_base, outputs=set_status)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7862)
