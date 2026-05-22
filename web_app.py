#!/usr/bin/env python3
"""Minimal web UI for the facial acne RAG assistant."""

from __future__ import annotations

import json
import os
import traceback
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.utils import secure_filename

from rag.pipeline import stream_rag_events

SCRIPT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = SCRIPT_DIR / "uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

load_dotenv(SCRIPT_DIR / ".env")

app = Flask(__name__, template_folder=str(SCRIPT_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask/stream", methods=["POST"])
def api_ask_stream():
    question = (request.form.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    use_rag = request.form.get("use_rag", "true").lower() in ("true", "1", "on", "yes")
    chroma_path = Path(os.environ.get("CHROMA_PATH", SCRIPT_DIR / "chroma_db"))
    collection = os.environ.get("COLLECTION_NAME", "acne_research")

    if use_rag and not chroma_path.is_dir():
        return jsonify(
            {
                "error": "Chroma database not found. Run: python ingest_papers.py --reset",
            }
        ), 503

    image_path: Path | None = None
    upload = request.files.get("image")
    if upload and upload.filename:
        if not _allowed_file(upload.filename):
            return jsonify({"error": "Image must be JPG, PNG, or WebP."}), 400
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        ext = Path(upload.filename).suffix.lower()
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        upload.save(dest)
        image_path = dest

    def generate():
        try:
            for event in stream_rag_events(
                question,
                image_path=image_path,
                chroma_path=chroma_path if use_rag else None,
                collection_name=collection,
                use_rag=use_rag,
                top_k=7,
            ):
                yield _sse(event)
        except Exception as exc:
            app.logger.error("stream failed: %s", traceback.format_exc())
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            if image_path and image_path.is_file():
                try:
                    image_path.unlink()
                except OSError:
                    pass

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the acne RAG web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Open http://{args.host}:{args.port}")
    print("Requires LM Studio (chat + embed) and GEMINI_API_KEY for photos.")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
