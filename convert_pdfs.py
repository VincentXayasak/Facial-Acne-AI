#!/usr/bin/env python3
"""
Batch-convert academic PDFs to Markdown using Marker (datalab-to/marker).

Defaults are tuned for RAG: markdown output, per-document metadata, resume via
--skip-existing, and an index manifest for downstream ingestion.

Install:  pip install -r requirements.txt
Usage:    python convert_pdfs.py
          python convert_pdfs.py --force-ocr --workers 2
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Marker/surya multiprocessing and thread settings (from marker.scripts.convert)
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("IN_STREAMLIT", "true")

import psutil
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from marker.config.parser import ConfigParser
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import output_exists, save_output, text_from_rendered
from marker.utils.batch import get_batch_sizes_worker_counts
from marker.utils.gpu import GPUManager

configure_logging()
logger = get_logger()

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "pdfs"
DEFAULT_OUTPUT = SCRIPT_DIR / "markdown"
PDF_SUFFIXES = {".pdf", ".PDF"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert academic PDFs to clean Markdown with Marker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="Folder containing PDF files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Root folder for converted markdown and metadata",
    )
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json", "html", "chunks"],
        default="markdown",
        help="Marker output format ('chunks' is flattened blocks, good for RAG)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes (Marker picks a default if omitted)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process at most this many PDFs (useful for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs that already have output in the output folder",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Copy .md/.json and _meta.json into the output root (no per-PDF subfolders)",
    )
    parser.add_argument(
        "--paginate-output",
        action="store_true",
        help="Insert page breaks in markdown (\\n\\n{PAGE_NUMBER} + rule)",
    )
    parser.add_argument(
        "--disable-image-extraction",
        action="store_true",
        help="Do not extract images (smaller corpus for text-only RAG)",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="OCR all lines; improves inline math and scanned papers",
    )
    parser.add_argument(
        "--strip-existing-ocr",
        action="store_true",
        help="Remove embedded OCR text and re-OCR with Surya",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use an LLM backend for higher accuracy (requires API keys)",
    )
    parser.add_argument(
        "--llm-service",
        type=str,
        default=None,
        help="Full import path for LLM service (default: Google Gemini)",
    )
    parser.add_argument(
        "--page-range",
        type=str,
        default=None,
        help='Pages to convert, e.g. "0,5-10,20"',
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Extra Marker config JSON (see: marker_single --help)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path for ingestion manifest (default: <output-dir>/manifest.json)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Marker debug output",
    )
    return parser.parse_args()


def discover_pdfs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    pdfs = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix in PDF_SUFFIXES
    )
    if not pdfs:
        logger.warning("No PDF files found in %s", input_dir)
    return pdfs


def title_from_metadata(metadata: dict) -> str | None:
    toc = metadata.get("table_of_contents") or []
    if not toc:
        return None
    first = toc[0]
    if isinstance(first, dict):
        return first.get("title")
    return None


def flatten_outputs(output_dir: Path, base_name: str, ext: str) -> None:
    """Copy primary output and metadata to the output root."""
    sub = output_dir / base_name
    if not sub.is_dir():
        return
    for name in (f"{base_name}.{ext}", f"{base_name}_meta.json"):
        src = sub / name
        if src.is_file():
            dst = output_dir / name
            dst.write_bytes(src.read_bytes())


# --- multiprocessing workers (models loaded once per worker) ---

model_refs: dict | None = None


def worker_init() -> None:
    global model_refs
    model_refs = create_model_dict()
    atexit.register(worker_exit)


def worker_exit() -> None:
    global model_refs
    try:
        del model_refs
    except Exception:
        pass


def build_cli_options(args: argparse.Namespace) -> dict:
    opts: dict = {
        "output_dir": str(args.output_dir.resolve()),
        "output_format": args.output_format,
        "skip_existing": args.skip_existing,
        "disable_multiprocessing": True,
        "debug": args.debug,
    }
    if args.workers is not None:
        opts["workers"] = args.workers
    if args.paginate_output:
        opts["paginate_output"] = True
    if args.disable_image_extraction:
        opts["disable_image_extraction"] = True
    if args.force_ocr:
        opts["force_ocr"] = True
    if args.strip_existing_ocr:
        opts["strip_existing_ocr"] = True
    if args.use_llm:
        opts["use_llm"] = True
    if args.llm_service:
        opts["llm_service"] = args.llm_service
    if args.page_range:
        opts["page_range"] = args.page_range
    if args.config_json:
        opts["config_json"] = str(args.config_json.resolve())
    return opts


def process_single_pdf(task: tuple[Path, dict]) -> dict:
    """Convert one PDF; return a manifest entry dict."""
    pdf_path, cli_options = task
    pdf_path = pdf_path.resolve()
    torch.set_num_threads(cli_options["total_torch_threads"])

    local_opts = {k: v for k, v in cli_options.items() if k != "total_torch_threads"}
    flat_output = local_opts.pop("flat_output", False)

    config_parser = ConfigParser(local_opts)
    out_folder = config_parser.get_output_folder(str(pdf_path))
    base_name = config_parser.get_base_filename(str(pdf_path))

    entry: dict = {
        "pdf": str(pdf_path),
        "base_name": base_name,
        "status": "pending",
        "pages": 0,
        "title": None,
        "markdown_path": None,
        "metadata_path": None,
        "error": None,
    }

    if local_opts.get("skip_existing") and output_exists(out_folder, base_name):
        entry["status"] = "skipped"
        meta_path = Path(out_folder) / f"{base_name}_meta.json"
        if meta_path.is_file():
            entry["metadata_path"] = str(meta_path)
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                entry["title"] = title_from_metadata(meta)
            except json.JSONDecodeError:
                pass
        ext = {"markdown": "md", "html": "html", "json": "json", "chunks": "json"}[
            local_opts["output_format"]
        ]
        entry["markdown_path"] = str(Path(out_folder) / f"{base_name}.{ext}")
        return entry

    converter_cls = config_parser.get_converter_cls()
    config_dict = config_parser.generate_config_dict()
    config_dict["disable_tqdm"] = True

    try:
        converter = converter_cls(
            config=config_dict,
            artifact_dict=model_refs,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(str(pdf_path))
        save_output(rendered, out_folder, base_name)

        text, ext, _images = text_from_rendered(rendered)
        entry["pages"] = getattr(converter, "page_count", 0) or 0
        entry["status"] = "ok"
        entry["markdown_path"] = str(Path(out_folder) / f"{base_name}.{ext}")
        entry["metadata_path"] = str(Path(out_folder) / f"{base_name}_meta.json")
        entry["title"] = title_from_metadata(rendered.metadata or {})

        if flat_output:
            flatten_outputs(Path(local_opts["output_dir"]), base_name, ext)

        del rendered
        del converter
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = str(exc)
        logger.error("Failed to convert %s: %s", pdf_path.name, exc)
        traceback.print_exc()
    finally:
        gc.collect()

    return entry


def write_manifest(path: Path, args: argparse.Namespace, entries: list[dict], elapsed: float) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "output_format": args.output_format,
        "elapsed_seconds": round(elapsed, 2),
        "summary": {
            "total": len(entries),
            "ok": sum(1 for e in entries if e["status"] == "ok"),
            "skipped": sum(1 for e in entries if e["status"] == "skipped"),
            "error": sum(1 for e in entries if e["status"] == "error"),
        },
        "documents": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote manifest: %s", path)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = discover_pdfs(args.input_dir)
    if args.max_files:
        pdfs = pdfs[: args.max_files]
    if not pdfs:
        return 1

    cli_options = build_cli_options(args)
    cli_options["flat_output"] = args.flat_output

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    manifest_path = args.manifest or (args.output_dir / "manifest.json")

    print(f"Found {len(pdfs)} PDF(s) in {args.input_dir}")
    print(f"Writing output to {args.output_dir}")
    if args.skip_existing:
        print("Skipping PDFs that already have output (--skip-existing)")

    start = time.time()
    entries: list[dict] = []
    total_pages = 0

    with GPUManager(0) as gpu_manager:
        batch_sizes, workers = get_batch_sizes_worker_counts(gpu_manager, 7)
        if args.workers is not None:
            workers = args.workers
        cli_options.update(batch_sizes)

        total_processes = max(1, min(len(pdfs), workers))
        cli_options["total_torch_threads"] = max(
            2, psutil.cpu_count(logical=False) // total_processes
        )

        logger.info(
            "Converting %d PDFs with %d worker(s)",
            len(pdfs),
            total_processes,
        )

        tasks = [(p, cli_options) for p in pdfs]

        if total_processes == 1:
            worker_init()
            for task in tqdm(tasks, desc="Converting PDFs", unit="pdf"):
                entry = process_single_pdf(task)
                entries.append(entry)
                if entry["status"] in ("ok", "skipped"):
                    total_pages += entry.get("pages", 0)
        else:
            with mp.Pool(
                processes=total_processes,
                initializer=worker_init,
                maxtasksperchild=10,
            ) as pool:
                for entry in tqdm(
                    pool.imap_unordered(process_single_pdf, tasks),
                    total=len(tasks),
                    desc="Converting PDFs",
                    unit="pdf",
                ):
                    entries.append(entry)
                    if entry["status"] in ("ok", "skipped"):
                        total_pages += entry.get("pages", 0)

    elapsed = time.time() - start
    entries.sort(key=lambda e: e["pdf"])

    write_manifest(manifest_path, args, entries, elapsed)

    ok = sum(1 for e in entries if e["status"] == "ok")
    skipped = sum(1 for e in entries if e["status"] == "skipped")
    failed = sum(1 for e in entries if e["status"] == "error")

    print()
    print(f"Done in {elapsed:.1f}s — {ok} converted, {skipped} skipped, {failed} failed")
    if total_pages and elapsed > 0:
        print(f"Throughput: {total_pages / elapsed:.2f} pages/sec")
    print(f"Manifest: {manifest_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
