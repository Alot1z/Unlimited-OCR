# GTX 1080 Ti / Pascal compatibility

**Status: experimental investigation; end-to-end inference is not yet verified.**

This guide records the current compatibility boundary for a Windows machine with an NVIDIA GTX 1080 Ti (Pascal, compute capability 6.1, 11 GiB VRAM). It does not claim that Unlimited-OCR-2 currently runs on this GPU.

## Findings

1. The repository's reference Transformers setup was tested on an A800 80 GB and loads the model as BF16. Pascal GPUs do not provide native BF16 support, so the documented BF16 load command is not a valid GTX 1080 Ti recipe.
2. The model has approximately 3.344 billion parameters. FP16 parameter storage alone is about **6.23 GiB**. This excludes the vision encoder's intermediate activations, generation state, temporary attention tensors, CUDA context, and other allocations. An 11 GiB card therefore has limited and unproven headroom.
3. The exact model revision audited is `c14071b549f4fcab10909f3c4acc6a2edb366947`. Its config enables `dash_use_triton=true`, and its remote code contains two explicit pure-PyTorch fallbacks: DASH attention uses chunked `scaled_dot_product_attention`, while the base sparse path uses PyTorch matmul/softmax when its fused kernel is disabled or unavailable. These are source-confirmed fallbacks, not yet GPU-verified.
4. Pascal cannot run the model's Triton kernels. The source exposes the following switches to force its non-Triton path: config `sparse_attention_config.dash_use_triton=false`, environment `GATHER_ATTN_TRITON=0`, and environment `INDEXER_SCORE_TRITON=0`. Keep `use_flex_attention=false` and `use_xformers=false` so the fallback stays on ordinary PyTorch operations. The dense fallbacks can use more memory and time than the fused kernels.
5. The repository's `infer.py` is an HTTP client for an SGLang server. It is not a local Transformers inference runner. The model's `_uocr_build_inputs` helper also hardcodes its image tensors to BF16; the Pascal runner must cast both image batches to FP16 before forwarding them to the FP16 model.

## Run the local preflight

The preflight does not download model weights, execute Hugging Face remote code, or send document images anywhere.

```powershell
.\.venv\Scripts\python.exe -m compat.gtx1080ti
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

Do not install Triton expecting it to enable the fused sparse-attention path on Pascal. The local runner explicitly disables the three discovered Triton paths and converts the model's hard-coded BF16 image inputs to FP16. The resulting pure-PyTorch fallback is still experimental and may be too slow or memory-intensive for longer pages.

## Local Pascal inference candidate

The repository now includes a Windows-first candidate runner and setup file:

```powershell
.\scripts\setup_gtx1080ti.ps1
.\.venv\Scripts\python.exe -m compat.local_runner --input .\path\to\document.pdf --max-pages 1
```

The runner pins the model revision above, loads FP16, disables Triton/flex-attention/xFormers paths, and patches the remote model's image-builder output tensors from BF16 to FP16 in memory. It does not modify the Hugging Face cache. PDF support uses PyMuPDF. The default is one page and 2,048 output tokens to limit the first test's memory use.

## What must be proven before a supported recipe can be published

- [x] Review the exact pinned Hugging Face model revision and direct/indirect custom-code imports.
- [x] Trace the active sparse-attention dispatch and identify the pure-PyTorch DASH and base-attention fallbacks.
- [ ] Run the fallback path on the actual GTX 1080 Ti with `dash_use_triton=false`, `GATHER_ATTN_TRITON=0`, and `INDEXER_SCORE_TRITON=0`.
- [ ] Load the model with FP16 and the in-memory image dtype patch on the target GPU; record peak allocated/reserved VRAM.
- [ ] Run a one-page image and a short PDF end to end; verify output quality and page markers.
- [ ] Run a multi-page case with bounded page count and output tokens; record peak VRAM, latency, and failures.
- [ ] Confirm the same scripts still preserve the original SGLang benchmark-client workflow.

Until the real one-page and multi-page tests pass on a GTX 1080 Ti, the truthful status remains **EXPERIMENTAL / NOT VERIFIED**. The source-level fallback is available, but end-to-end hardware compatibility is not yet established.

## Sources

- [Repository inference and tested A800/BF16 setup](https://github.com/Alot1z/Unlimited-OCR/blob/main/README.md)
- [Unlimited-OCR-2 model files](https://huggingface.co/baidu/Unlimited-OCR-2/tree/main)
- [Triton Windows GPU compatibility matrix](https://github.com/triton-lang/triton-windows)
- [Upstream Triton supported hardware](https://github.com/triton-lang/triton)
- [PyTorch previous-version installation commands](https://pytorch.org/get-started/previous-versions/)
- [Hugging Face guidance for custom model code](https://huggingface.co/docs/transformers/main/models)
