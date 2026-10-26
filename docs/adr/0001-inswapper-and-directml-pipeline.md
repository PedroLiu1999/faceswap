# 1. Inswapper 128 and DirectML Inference Pipeline

The real-time faceswap engine uses `inswapper_128.onnx` paired with `SCRFD` and `ArcFace`, running on Microsoft ONNX Runtime with DirectML as the default Windows execution provider, omitting per-frame face restoration models.

## Context

Real-time live video requires sustaining 30–60 FPS with minimal end-to-end latency (<33ms per frame). Higher-resolution swap models like SimSwap-512 and face restoration models like GFPGAN take 40–120ms per frame, causing unacceptable lag on consumer hardware. Furthermore, requiring CUDA restricts the application to NVIDIA users with the CUDA toolkit installed.

## Decision

1. Use `inswapper_128.onnx` on normalized $128 \times 128$ crops combined with SCRFD (which detects faces and extracts 5-point landmarks in a single pass).
2. Cache the 512-dimensional ArcFace embedding of the Target Face in memory upon upload to eliminate per-frame recognition overhead.
3. Composite swapped faces back into the original camera resolution using inverse affine transformation and feathered edge blending instead of running neural face enhancers.
4. Default to DirectML via `ort` on Windows for zero-configuration DirectX 12 hardware acceleration across AMD, Intel, and NVIDIA GPUs, while providing an opt-in `cuda` feature for NVIDIA RTX enthusiasts.
