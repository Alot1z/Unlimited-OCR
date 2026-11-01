"""Experimental FP16 runner using the pinned model's pure-PyTorch attention fallbacks."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "baidu/Unlimited-OCR-2"
DEFAULT_REVISION = "c14071b549f4fcab10909f3c4acc6a2edb366947"


def apply_pascal_runtime_overrides(config: Any, environ: dict[str, str] | None = None) -> Any:
    env = os.environ if environ is None else environ
    sparse = getattr(config, "sparse_attention_config", None)
    if not isinstance(sparse, dict):
        raise RuntimeError("Model config has no sparse_attention_config dictionary")
    sparse["dash_use_triton"] = False
    sparse["use_flex_attention"] = False
    sparse["use_xformers"] = False
    env["GATHER_ATTN_TRITON"] = "0"
    env["INDEXER_SCORE_TRITON"] = "0"
    return config


def cast_image_inputs(result: Any, dtype: Any) -> Any:
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        raise ValueError("_uocr_build_inputs must return four values")
    input_ids, images, seq_mask, spatial = result
    if images is None:
        return result
    if not isinstance(images, (tuple, list)) or len(images) != 2:
        raise ValueError("Expected crop and global image tensors")
    return input_ids, tuple(image.to(dtype=dtype) for image in images), seq_mask, spatial


def patch_image_builder(model: Any, dtype: Any) -> Any:
    module = sys.modules.get(model.__class__.__module__)
    if module is None:
        raise RuntimeError("Loaded Hugging Face model module was not found")
    if getattr(module, "_pascal_fp16_image_patch", False):
        return module
    original = getattr(module, "_uocr_build_inputs", None)
    if not callable(original):
        raise RuntimeError("Pinned model revision no longer exposes _uocr_build_inputs")

    def patched(*args: Any, **kwargs: Any) -> Any:
        return cast_image_inputs(original(*args, **kwargs), dtype)

    module._uocr_build_inputs = patched
    module._pascal_fp16_image_patch = True
    return module


def _load_pages(paths: list[str], max_pages: int, dpi: int) -> list[Any]:
    from PIL import Image

    pages = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".pdf":
            try:
                import fitz
            except ImportError as exc:
                raise RuntimeError("PDF input requires PyMuPDF from requirements-gtx1080ti.txt") from exc
            with fitz.open(path) as pdf:
                for index in range(min(len(pdf), max_pages - len(pages))):
                    pix = pdf.load_page(index).get_pixmap(
                        matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False
                    )
                    pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
                    if len(pages) >= max_pages:
                        break
        else:
            with Image.open(path) as image:
                pages.append(image.convert("RGB"))
        if len(pages) >= max_pages:
            break
    if not pages:
        raise RuntimeError("No input pages were loaded")
    return pages


def _load_model(model_id: str, revision: str, torch: Any) -> tuple[Any, Any]:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    apply_pascal_runtime_overrides(config)
    model = AutoModel.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, config=config,
        torch_dtype=torch.float16, use_safetensors=True,
    ).eval().to("cuda")
    patch_image_builder(model, torch.float16)
    routers = [m for m in model.modules() if m.__class__.__name__ == "DashAttention"]
    if not routers or any(getattr(router, "use_triton", True) for router in routers):
        raise RuntimeError("DASH routers are missing or Triton remains enabled")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    return model, tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental Unlimited-OCR-2 FP16 runner for Pascal sm_61.")
    parser.add_argument("--input", nargs="+", required=True, help="Image/PDF paths in page order")
    parser.add_argument("--output", help="Markdown output path")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Reviewed Hugging Face commit SHA")
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-page-tokens", type=int, default=2048)
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--dash-topm", type=int, default=16)
    parser.add_argument("--prompt", default="Multi page parsing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_pages < 1 or args.dpi < 36:
        raise SystemExit("--max-pages must be positive and --dpi must be at least 36")
    # These must be set before Transformers imports the model's dynamic code.
    os.environ["GATHER_ATTN_TRITON"] = "0"
    os.environ["INDEXER_SCORE_TRITON"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from compat.gtx1080ti import run_probe

    report = run_probe(torch)
    print(json.dumps(report, indent=2))
    if report.get("status") == "BLOCKED":
        return 2
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (6, 1):
        print(f"This runner targets Pascal sm_61; detected {capability}.", file=sys.stderr)
        return 2

    pages = _load_pages(args.input, args.max_pages, args.dpi)
    print(f"Loaded {len(pages)} page(s); model remote code is pinned to {args.revision}.")
    try:
        model, tokenizer = _load_model(args.model_id, args.revision, torch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("Model weights did not fit in free VRAM.", file=sys.stderr)
        return 3

    allocated_after_load = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = model.infer(
            tokenizer, prompt=args.prompt, image_file=pages, crop_mode=False,
            base_size=args.base_size, image_size=args.image_size, max_crops=1,
            max_length=args.max_length, max_new_tokens=args.max_new_tokens,
            max_page_tokens=args.max_page_tokens, dash_topm=args.dash_topm,
            eval_mode=True, save_results=False, return_dict=True,
        )
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("Inference ran out of VRAM; retry with one page and lower token limits.", file=sys.stderr)
        return 3

    elapsed = time.perf_counter() - started
    text = result.get("text", "") if isinstance(result, dict) else str(result)
    token_ids = result.get("token_ids", []) if isinstance(result, dict) else []
    first = Path(args.input[0])
    output = Path(args.output) if args.output else Path("outputs") / f"{first.stem}_ocr.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    metadata = {
        "model_id": args.model_id, "model_revision": args.revision,
        "gpu": torch.cuda.get_device_name(), "compute_capability": "6.1",
        "dtype": "float16", "pages": len(pages), "generated_tokens": len(token_ids),
        "elapsed_seconds": round(elapsed, 2),
        "allocated_after_load_gib": round(allocated_after_load / (1024 ** 3), 2),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2),
        "fallbacks": {"dash_use_triton": False, "GATHER_ATTN_TRITON": "0",
                      "INDEXER_SCORE_TRITON": "0", "use_flex_attention": False,
                      "use_xformers": False},
        "model_inference_verified_for_this_run": True,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"OCR output: {output.resolve()}")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
