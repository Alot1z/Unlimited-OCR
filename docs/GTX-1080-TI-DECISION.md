# Decision record: Pascal inference support

- **ID:** UOCR-GPU-001
- **Status:** active
- **Decision:** Do not describe Unlimited-OCR-2 as supported on Pascal / GTX 1080 Ti until the FP16 + pure-PyTorch fallback path is exercised end to end on the target GPU.
- **Why:** The repository's documented path is BF16 and tested on an A800 80 GB. Pascal cannot run the model's Triton kernels, but the pinned source revision contains pure-PyTorch dense fallbacks controlled by `dash_use_triton`, `GATHER_ATTN_TRITON`, and `INDEXER_SCORE_TRITON`. The model also hardcodes BF16 image tensors, requiring an in-memory FP16 input-builder patch. FP16 parameter storage is estimated at 6.23 GiB before runtime allocations.
- **Evidence:** Repository README and inference script; Hugging Face Unlimited-OCR-2 model files and configuration; Triton Windows hardware compatibility matrix; parameter-count metadata.
- **Confidence:** The source-level fallback switches and BF16 image-tensor behavior are confirmed at model revision `c14071b549f4fcab10909f3c4acc6a2edb366947`. Whether the fallback is stable and fits in 11 GiB on Windows/Pascal remains unverified.
- **Alternatives:** The candidate is FP16 with the PyTorch dense fallback, not a new backend or quantized checkpoint. The fallback may be slower and more memory-intensive.
- **Consequences:** Keep the original SGLang workflow intact. Provide a non-invasive preflight and a hardware-validation checklist. Only promote the status after a real one-page and multi-page inference run succeeds on Pascal.
- **Next action:** Run the pinned local runner on a synthetic one-page document, then a short multi-page PDF; record output quality, peak VRAM, latency, and any fallback errors.

- **Prototype pointer:** `prototype/gtx1080ti-compat-gate` → `prototypes/gtx1080ti-compatibility-prototype.html` (throwaway state exploration; not production code).
- **Tracking limitation:** GitHub Issues are disabled for this repository, so no issue could be created for the prototype handoff.
