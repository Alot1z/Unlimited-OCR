# GTX 1080 Ti / Pascal compatibility

**Status: experimental investigation; end-to-end inference is not yet verified.**

This guide records the current compatibility boundary for a Windows machine with an NVIDIA GTX 1080 Ti (Pascal, compute capability 6.1, 11 GiB VRAM). It does not claim that Unlimited-OCR-2 currently runs on this GPU.

## Findings

1. The repository's reference Transformers setup was tested on an A800 80 GB and loads the model as BF16. Pascal GPUs do not provide native BF16 support, so the documented BF16 load command is not a valid GTX 1080 Ti recipe.
2. The model has approximately 3.344 billion parameters. FP16 parameter storage alone is about **6.23 GiB**. This excludes the vision encoder's intermediate activations, generation state, temporary attention tensors, CUDA context, and other allocations. An 11 GiB card therefore has limited and unproven headroom.
3. The Hugging Face model repository contains custom sparse-attention/Triton source files (`sparse_gather_triton.py` and `sparse_gather_e2e_triton.py`) and a configuration that enables sparse attention. The model's exact fallback behavior when those kernels cannot run must be audited and tested before claiming Pascal support.
4. The Triton Windows compatibility project explicitly lists consumer Pascal (`sm_61`, including GTX 10-series) as unsupported. The upstream Triton project also documents a newer supported NVIDIA architecture baseline. Merely installing Triton, or omitting Triton, does not prove that the model has a usable non-Triton path.
5. The repository's `infer.py` is an HTTP client for an SGLang server. It is not a local Transformers inference runner. The custom SGLang wheel and server settings in the main README are not a demonstrated Pascal/Windows configuration.

## Run the local preflight

The preflight does not download model weights, execute Hugging Face remote code, or send document images anywhere.

```powershell
python -m compat.gtx1080ti
```

It reports:

- whether PyTorch can see a CUDA device;
- GPU name, compute capability, free/total VRAM, and PyTorch's compiled CUDA architecture targets;
- whether PyTorch reports BF16 support;
- whether a tiny FP16 CUDA matrix operation succeeds;
- whether Triton is installed and whether the model's known Triton path is a compatibility blocker;
- the FP16 parameter-storage estimate.

A successful FP16 matrix operation is only a basic CUDA check. It is **not** proof that the custom OCR attention kernels or the complete model work.

Exit code `0` means the static checks found no known blocker and a real model smoke test is still required. Exit code `2` means the preflight found a blocker or could not access CUDA.

## Candidate software stack

The upstream repository's tested `torch==2.13.0` BF16/A800 recipe must not be copied to Pascal unchanged. Older official Windows CUDA wheels are available, but the installed wheel must be checked at runtime rather than assumed compatible.

For any candidate PyTorch installation, verify the installed build first:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_arch_list() if torch.cuda.is_available() else [])"
```

The preflight accepts a same-major lower-minor cubin target where CUDA binary compatibility permits it (for example, `sm_60` on `sm_61`). The actual FP16 smoke test is the final check of basic kernel execution.

Do not install a current Triton package expecting it to enable the model's sparse-attention path on Pascal. Do not switch the model to FP16 and infer that all remaining custom kernels are compatible.

## What must be proven before a supported recipe can be published

- [ ] Review the exact pinned Hugging Face model revision and all direct/indirect custom-code imports.
- [ ] Trace the active sparse-attention dispatch and determine whether `_sparse_base_impl.py` is a real non-Triton fallback on Pascal.
- [ ] If a fallback exists, test it explicitly with sparse attention enabled and disabled where supported; do not patch model config speculatively.
- [ ] Load the model with FP16 on the target GPU and record peak allocated/reserved VRAM.
- [ ] Run a one-page image and a short PDF end to end; verify output quality and page markers.
- [ ] Run a multi-page case with bounded page count and output tokens; record peak VRAM, latency, and failures.
- [ ] Confirm the same scripts still preserve the original SGLang benchmark-client workflow.

Until these checks pass on a real GTX 1080 Ti, the truthful status remains **BLOCKED / NOT VERIFIED**, not supported.

## Sources

- [Repository inference and tested A800/BF16 setup](https://github.com/Alot1z/Unlimited-OCR/blob/main/README.md)
- [Unlimited-OCR-2 model files](https://huggingface.co/baidu/Unlimited-OCR-2/tree/main)
- [Triton Windows GPU compatibility matrix](https://github.com/triton-lang/triton-windows)
- [Upstream Triton supported hardware](https://github.com/triton-lang/triton)
- [PyTorch previous-version installation commands](https://pytorch.org/get-started/previous-versions/)
- [Hugging Face guidance for custom model code](https://huggingface.co/docs/transformers/main/models)
