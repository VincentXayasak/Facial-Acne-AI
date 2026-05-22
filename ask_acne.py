#!/usr/bin/env python3
"""
Facial acne RAG assistant: Gemini vision → Chroma retrieval → Llama 3.2 (LM Studio).

Examples:
  # Text-only (no photo)
  python ask_acne.py --question "What does research say about benzoyl peroxide for inflammatory acne?"

  # With photo (Gemini vision → RAG → Llama)
  python ask_acne.py --image photo.jpg --question "What might help based on what you see and the literature?"

  # Base Llama only (Gemini observation still used if --image); no Chroma retrieval
  python ask_acne.py --no-rag --image photo.jpg --question "What might help for what you see?"

  # Skip vision; use a saved observation JSON
  python ask_acne.py --question "Treatment options?" --observation-json observation.json

  # Health checks
  python ask_acne.py --ping-chat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag.embeddings import LMStudioEmbeddingFunction
from rag.llm import LMStudioChat
from rag.pipeline import print_result, result_to_dict, run_rag

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask acne research questions with optional face photo + RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="User question for Llama",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=Path,
        default=None,
        help="Face photo path (sent to Gemini for structured observation)",
    )
    parser.add_argument(
        "--observation-json",
        type=Path,
        default=None,
        help="Skip Gemini; use this JSON observation file",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Use base Llama only (no Chroma); Gemini vision still runs if --image is set",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Ignore --image for Gemini (Llama only)",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=None,
        help="Chroma directory (default: CHROMA_PATH or ./chroma_db)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection (default: COLLECTION_NAME or acne_research)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=7,
        help="Unique paper-section chunks to retrieve (higher = richer context)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full result as JSON",
    )
    parser.add_argument(
        "--ping-chat",
        action="store_true",
        help="Test LM Studio chat model and exit",
    )
    parser.add_argument(
        "--ping-embed",
        action="store_true",
        help="Test LM Studio embedding model and exit",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(SCRIPT_DIR / ".env")
    args = parse_args()

    chroma_path = args.chroma_path or Path(
        os.environ.get("CHROMA_PATH", SCRIPT_DIR / "chroma_db")
    )
    collection = args.collection or os.environ.get("COLLECTION_NAME", "acne_research")

    if args.ping_embed:
        fn = LMStudioEmbeddingFunction()
        try:
            dim = fn.ping()
            print(f"Embeddings OK — {fn.model} @ {fn.base_url} (dim={dim})")
            return 0
        except Exception as exc:
            print(f"Embeddings FAILED: {exc}", file=sys.stderr)
            return 1

    if args.ping_chat:
        chat = LMStudioChat()
        try:
            reply = chat.ping()
            print(f"Chat OK — {chat.model} @ {chat.base_url}")
            print(f"Reply: {reply!r}")
            return 0
        except Exception as exc:
            print(f"Chat FAILED: {exc}", file=sys.stderr)
            print(
                "Load Llama 3.2 8B Instruct in LM Studio and set LM_STUDIO_CHAT_MODEL to its id.",
                file=sys.stderr,
            )
            return 1

    if not args.question:
        print("Provide --question (or use --ping-chat / --ping-embed).", file=sys.stderr)
        return 1

    use_rag = not args.no_rag
    if use_rag and not chroma_path.is_dir():
        print(
            f"Chroma DB not found at {chroma_path}. Run: python ingest_papers.py --reset",
            file=sys.stderr,
        )
        return 1

    observation = None
    if args.observation_json:
        observation = json.loads(
            args.observation_json.read_text(encoding="utf-8")
        )

    image_path = args.image
    if image_path and not image_path.is_file():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    try:
        result = run_rag(
            args.question,
            image_path=image_path if not args.skip_vision else None,
            observation=observation,
            chroma_path=chroma_path if use_rag else None,
            collection_name=collection,
            top_k=args.top_k,
            skip_vision=args.skip_vision,
            use_rag=use_rag,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result_to_dict(result), indent=2, ensure_ascii=False))
    else:
        print_result(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
