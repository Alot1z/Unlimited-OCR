"""Conservative hardware preflight for local Unlimited-OCR inference.

This module does not download weights or execute Hugging Face remote model code.
It reports basic CUDA capability separately from model-kernel compatibility.
"""
from __future__ import annotations

from typing import Any

MODEL_PARAMETERS = 3_344_000_000
GIB = 1024 ** 3


def estimate_weight_memory_gib(
    parameters: int = MODEL_PARAMETERS, dtype: str = "float16"
) -> float:
    """Estimate raw parameter storage; excludes all runtime allocations."""
    if not isinstance(parameters, int) or parameters < 0:
        raise ValueError("parameters must be a non-negative integer")
    sizes = {"float16": 2, "bfloat16": 2, "float32": 4}
    if dtype not in sizes:
        raise ValueError("dtype must be float16, bfloat16, or float32")
    return parameters * sizes[dtype] / GIB


def _normalise_architectures(architectures: list[str] | tuple[str, ...] | None) -> set[str]:
    return {str(item).lower().replace("compute_", "sm_") for item in (architectures or [])}


def assess_device(
    *,
    name: str,
    capability: tuple[int, int],
    total_vram_gib: float,
    free_vram_gib: float,
    architectures: list[str] | tuple[str, ...] | None,
    bf16_supported: bool,
    triton_installed: bool,
    model_uses_triton_kernels: bool = True,
    non_triton_fallback_available: bool = True,
) -> dict[str, Any]:
    """Assess known static risks without claiming end-to-end model support."""
    major, minor = capability
    sm = f"sm_{major}{minor}"
    compiled = _normalise_architectures(architectures)
    blockers: list[str] = []
    findings: list[dict[str, str]] = []

    # CUDA cubins are compatible within the same compute-capability major:
    # sm_60 can execute on sm_61. Never accept a target from another major.
    has_compatible_arch = any(
        arch.startswith(f"sm_{major}")
        and arch[3:].isdigit()
        and int(arch[3:]) <= int(f"{major}{minor}")
        for arch in compiled
    )
    if not has_compatible_arch:
        blockers.append("PYTORCH_ARCH_MISSING")
        findings.append({
            "level": "error",
            "code": "PYTORCH_ARCH_MISSING",
            "message": f"Installed PyTorch reports no compatible {sm} kernel target.",
        })
    else:
        findings.append({
            "level": "ok",
            "code": "PYTORCH_ARCH_PRESENT",
            "message": f"Installed PyTorch reports a compatible target for {sm}; a CUDA smoke test is still required.",
        })

    if not bf16_supported:
        findings.append({
            "level": "warning",
            "code": "BF16_UNAVAILABLE",
            "message": "BF16 is not reported as supported. The documented BF16 load path is unsuitable; FP16 is the local candidate.",
        })

    triton_compatibility = "not-applicable"
    if model_uses_triton_kernels and (major, minor) < (7, 0):
        triton_compatibility = (
            "unsupported-architecture-fallback-available"
            if non_triton_fallback_available else "unsupported-architecture"
        )
        if non_triton_fallback_available:
            findings.append({
                "level": "warning",
                "code": "TRITON_FALLBACK_AVAILABLE_UNVERIFIED",
                "message": "Pascal cannot run the model's Triton kernels, but the pinned source has pure-PyTorch dense fallbacks. The local runner disables DASH, gather, and indexer Triton paths; the fallback still needs a real model smoke test and may use more memory/time.",
            })
        else:
            blockers.append("TRITON_ARCH_UNSUPPORTED")
            findings.append({
                "level": "error",
                "code": "TRITON_ARCH_UNSUPPORTED",
                "message": "The model requires Triton kernels on this architecture and no non-Triton fallback is available.",
            })
    elif model_uses_triton_kernels:
        triton_compatibility = "architecture-not-flagged"
        findings.append({
            "level": "warning",
            "code": "TRITON_RUNTIME_UNVERIFIED",
            "message": "Triton kernels are part of the model source; a real end-to-end model smoke test is still required.",
        })

    fp16_weights = estimate_weight_memory_gib(dtype="float16")
    if free_vram_gib < fp16_weights:
        blockers.append("INSUFFICIENT_FREE_VRAM_FOR_WEIGHTS")
        findings.append({
            "level": "error",
            "code": "INSUFFICIENT_FREE_VRAM_FOR_WEIGHTS",
            "message": f"{free_vram_gib:.2f} GiB is free; FP16 parameter storage alone is approximately {fp16_weights:.2f} GiB.",
        })
    else:
        findings.append({
            "level": "warning",
            "code": "VRAM_HEADROOM_UNPROVEN",
            "message": f"FP16 parameter storage is approximately {fp16_weights:.2f} GiB. Runtime headroom is not established.",
        })

    if model_uses_triton_kernels and not triton_installed:
        findings.append({
            "level": "info",
            "code": "TRITON_NOT_INSTALLED",
            "message": "Triton is absent; the Pascal runner disables the model's optional Triton paths and selects its source-level PyTorch fallback. This is not proof of successful inference.",
        })

    return {
        "device": name,
        "compute_capability": f"{major}.{minor}",
        "torch_arch_target": sm,
        "total_vram_gib": round(float(total_vram_gib), 2),
        "free_vram_gib": round(float(free_vram_gib), 2),
        "bf16_supported": bool(bf16_supported),
        "triton_installed": bool(triton_installed),
        "triton_compatibility": triton_compatibility,
        "non_triton_fallback_available": bool(non_triton_fallback_available),
        "fp16_parameter_storage_gib": round(fp16_weights, 2),
        "blockers": blockers,
        "findings": findings,
        "status": "BLOCKED" if blockers else "NEEDS_MODEL_SMOKE_TEST",
        "model_inference_verified": False,
    }


def run_probe(torch_module: Any | None = None) -> dict[str, Any]:
    """Probe the active CUDA device; torch is imported lazily."""
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError:
            return {
                "status": "BLOCKED",
                "blockers": ["PYTORCH_NOT_INSTALLED"],
                "findings": [{
                    "level": "error",
                    "code": "PYTORCH_NOT_INSTALLED",
                    "message": "PyTorch is not installed in this Python environment.",
                }],
                "model_inference_verified": False,
            }

    if not torch_module.cuda.is_available():
        return {
            "status": "BLOCKED",
            "blockers": ["CUDA_UNAVAILABLE"],
            "findings": [{
                "level": "error",
                "code": "CUDA_UNAVAILABLE",
                "message": "PyTorch cannot access a CUDA device. Check the NVIDIA driver and CUDA-enabled PyTorch build.",
            }],
            "torch_version": getattr(torch_module, "__version__", "unknown"),
            "torch_cuda_version": getattr(torch_module.version, "cuda", None),
            "model_inference_verified": False,
        }

    device_index = torch_module.cuda.current_device()
    props = torch_module.cuda.get_device_properties(device_index)
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(device_index)
    major, minor = torch_module.cuda.get_device_capability(device_index)
    try:
        bf16_supported = bool(torch_module.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        bf16_supported = False
    try:
        import triton  # noqa: F401
        triton_installed = True
    except Exception:
        triton_installed = False

    report = assess_device(
        name=str(props.name),
        capability=(int(major), int(minor)),
        total_vram_gib=total_bytes / GIB,
        free_vram_gib=free_bytes / GIB,
        architectures=torch_module.cuda.get_arch_list(),
        bf16_supported=bf16_supported,
        triton_installed=triton_installed,
        non_triton_fallback_available=True,
    )
    report["torch_version"] = getattr(torch_module, "__version__", "unknown")
    report["torch_cuda_version"] = getattr(torch_module.version, "cuda", None)

    # This tiny operation validates basic FP16 CUDA execution only. It does not
    # validate the OCR model's custom attention kernels or memory requirements.
    try:
        a = torch_module.ones(
            (16, 16), device=f"cuda:{device_index}", dtype=torch_module.float16
        )
        _ = a @ a
        torch_module.cuda.synchronize(device_index)
        report["fp16_cuda_smoke_test"] = "PASS"
    except Exception as exc:
        report["fp16_cuda_smoke_test"] = "FAIL"
        report["fp16_cuda_smoke_error"] = f"{type(exc).__name__}: {exc}"
        report["blockers"].append("FP16_CUDA_SMOKE_TEST_FAILED")
        report["status"] = "BLOCKED"
    return report


def main() -> int:
    import json

    report = run_probe()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") == "NEEDS_MODEL_SMOKE_TEST" else 2


if __name__ == "__main__":
    raise SystemExit(main())
