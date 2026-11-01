# Decision record: Pascal inference support

- **ID:** UOCR-GPU-001
- **Status:** active
- **Decision:** Do not describe Unlimited-OCR-2 as supported on Pascal / GTX 1080 Ti, and do not ship an FP16-only recipe, until the model's active sparse-attention dispatch and non-Triton fallback are verified on the target GPU.
- **Why:** The repository's documented path is BF16 and tested on an A800 80 GB. The model source includes Triton sparse-attention kernels; the Triton Windows compatibility matrix lists Pascal as unsupported. FP16 parameter storage is estimated at 6.23 GiB before runtime allocations, leaving the 11 GiB card with unproven headroom.
- **Evidence:** Repository README and inference script; Hugging Face Unlimited-OCR-2 model files and configuration; Triton Windows hardware compatibility matrix; parameter-count metadata.
- **Confidence:** The BF16 and Triton compatibility constraints are source-supported. Whether the model can select a usable non-Triton fallback is unknown until its full dispatch path is inspected and exercised.
- **Alternatives:** No alternative backend has been validated for this hardware. Do not infer that disabling Triton or changing dtype is sufficient.
- **Consequences:** Keep the original SGLang workflow intact. Provide a non-invasive preflight and a hardware-validation checklist. Only promote the status after a real one-page and multi-page inference run succeeds on Pascal.
- **Next action:** Inspect the exact model revision's attention dispatch and test any non-Triton fallback; then run the model on the target GTX 1080 Ti and record peak VRAM and output checks.

- **Prototype pointer:** `prototype/gtx1080ti-compat-gate` → `prototypes/gtx1080ti-compatibility-prototype.html` (throwaway state exploration; not production code).
- **Tracking limitation:** GitHub Issues are disabled for this repository, so no issue could be created for the prototype handoff.
