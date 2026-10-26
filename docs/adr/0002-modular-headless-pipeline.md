# 2. Modular Headless Pipeline Architecture

The faceswap engine isolates the vision and inference pipeline completely from the GUI and platform I/O devices.

## Context

Coupling the ONNX inference, image alignment, and temporal filtering directly to an `egui` UI or specific camera device drivers makes unit testing, benchmarking, and automated regression testing difficult.

## Decision

Structure the system into four decoupled modules:
1. `camera`: Input device enumeration, format negotiation, and frame polling.
2. `pipeline`: Model loading, face detection (`SCRFD`), Umeyama similarity transformation, ArcFace feature extraction, `inswapper_128` inference, One-Euro temporal filtering, and inverse affine blending.
3. `vcam`: Platform-specific virtual camera sinks with graceful fallback.
4. `gui`: Immediate-mode UI layout, file pickers, and viewport texture presentation.

The `pipeline` operates purely on abstract image buffers and math vectors, allowing automated testing and synthetic benchmarking without needing a physical webcam or GPU window context.
