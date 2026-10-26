## Problem Statement

Video callers, streamers, and remote workers who want to disguise their facial identity or project an alternate persona during live video conferencing (e.g., Discord, Zoom, Microsoft Teams, Google Meet) currently lack a fast, private, native desktop solution. Existing deepfake tools either run offline on recorded video, suffer from high latency and stuttering (failing to sustain 30 FPS), require complicated multi-gigabyte Python environments and CUDA toolkit installations, or lack a seamless virtual camera output on Windows.

## Solution

A high-performance native desktop application written in Rust that captures frames from a local physical webcam, detects the live face, swaps it in real time with a static target identity image chosen by the user, and streams the composited video feed directly into a virtual webcam device recognized by any meeting or streaming software. The application features an immediate-mode graphical interface with zero-configuration GPU acceleration via DirectML on Windows, smooth temporal jitter reduction, and automatic fallback to clean passthrough when no face is in view.

## User Stories

1. As a user, I want the application to automatically verify and download any missing ONNX model weights on initial launch, so that I don't have to manually locate, download, and configure model files.
2. As a user, I want to see a clear download progress bar in the GUI while models are being fetched, so that I know the application is actively downloading assets and has not hung.
3. As a user, I want to select my physical webcam from a dropdown list of available devices, so that I can choose which camera feed to capture.
4. As a user, I want the application to negotiate a default resolution of 1280x720 at 30 FPS with my webcam, so that I get a smooth, high-definition 16:9 feed without excessive GPU memory bandwidth.
5. As a user, I want to upload a Target Face photo (PNG, JPG, WebP) using a native file selection dialog, so that I can choose whose face to swap onto my live video.
6. As a user, I want the application to immediately validate the uploaded Target Face photo and display a clear error notification if no human face is detected, so that I don't stream corrupted or invalid frames.
7. As a user, I want to see a thumbnail preview of the active Target Face in the side control panel, so that I have visual confirmation of which identity is currently active.
8. As a user, I want to see a real-time live preview of the camera feed in the application window, so that I can inspect the face swap quality and framing before going live.
9. As a user, I want the face swap to achieve at least 30 frames per second on consumer GPUs, so that my video feed looks natural and fluid without perceptible lag.
10. As a user, I want the application to track and swap only the Primary Face when multiple people are visible in the camera frame, so that the frame rate remains stable and background individuals aren't randomly swapped.
11. As a user, I want face jitter and micro-vibrations around my eyes and chin to be smoothed out by a temporal filter, so that the swapped face remains steady even when I make subtle head movements.
12. As a user, I want the original camera background, lighting, and hair to remain sharp and unblurred around the swapped face, so that the swap looks natural without rectangular cropping artifacts.
13. As a user, I want the camera feed to cleanly pass through unmodified when no Target Face is loaded or when I step away from the camera, so that my virtual camera stream never freezes or goes black.
14. As a user, I want a single toggle button to start and stop broadcasting to the Virtual Camera Sink, so that I have complete control over when my video is exposed to other applications.
15. As a user, I want the application to stream to the OBS Virtual Camera transport by default, so that external apps like Zoom, Teams, Discord, and browsers recognize the virtual camera immediately.
16. As a user, I want the application to automatically fall back to the Windows 11 Media Foundation VCam if the OBS Virtual Camera driver is absent, so that it works seamlessly without extra setup on Windows 11.
17. As a user, I want the application to display a helpful banner with setup instructions if no virtual camera driver is installed on my system, while keeping the local GUI preview working, so that I can still preview the engine without crashes.
18. As a user, I want to see live performance metrics (current FPS and inference latency in milliseconds) in the control panel, so that I can monitor my system's real-time performance.
19. As a user, I want hardware acceleration to work out of the box on AMD, Intel, and NVIDIA graphics cards without needing to install the NVIDIA CUDA Toolkit, so that setup is zero-effort on any Windows machine.
20. As an advanced user with an NVIDIA RTX GPU, I want an optional build configuration enabling CUDA and TensorRT execution, so that I can unlock ultra-low latency sub-10ms inference.

## Implementation Decisions

- **Modular Seams**: The system is split into four decoupled modules:
  - `camera`: Manages webcam device enumeration, resolution/framerate negotiation, and non-blocking frame polling via Microsoft Media Foundation.
  - `pipeline`: Manages ONNX runtime sessions, face detection, Umeyama affine alignment, target embedding extraction, face swap neural inference, temporal filtering, and feathered inverse affine blending.
  - `vcam`: Wraps virtual camera sinks (OBS Virtual Camera and Windows 11 Media Foundation VCam) behind a unified trait, managing stream lifecycle and driver fallbacks.
  - `gui`: Builds the immediate-mode desktop window with responsive controls, native file dialogs, and dynamic GPU texture updates.
- **Model Pipeline & Inference**:
  - Face detection and 5-point landmark extraction run in a single pass using `SCRFD` (2.5G variant).
  - Target identity representation is extracted once using `ArcFace` (`w600k_r50`) and stored as a cached 512-dimensional vector in memory; per-frame face recognition is avoided.
  - Face swapping is performed using `inswapper_128.onnx` on normalized $128 \times 128$ face crops.
  - Inference executes via Microsoft ONNX Runtime using DirectML as the default Windows execution provider, with an optional compilation feature for CUDA.
  - Per-frame neural face restoration (GFPGAN/CodeFormer) is omitted to strictly preserve the 30+ FPS latency budget.
- **Image Mathematics & Compositing**:
  - Face alignment into the 128x128 canonical template is performed via the 2D Umeyama least-squares similarity transform (scale, rotation, translation).
  - A One-Euro temporal filter is applied across consecutive frames to smooth landmark points and affine transform parameters, eliminating micro-jitter.
  - Output blending reverses the affine transformation with bilinear interpolation and composites the swapped face into the native camera frame using an elliptical feathered alpha mask.
- **Concurrency Architecture**:
  - Camera capture, GPU inference, virtual camera publishing, and UI rendering execute on separate threads communicating via bounded lock-free channels.
  - When the inference worker is processing, the camera capture thread drops intermediate frames, ensuring the engine always works on the latest available frame (zero latency lag).
- **Graceful Fallbacks & Passthrough**:
  - When zero faces are detected or no target face is loaded, the pipeline performs Clean Passthrough, routing the original camera frame directly to the output.
  - If multiple faces are detected in a frame, the Primary Face (largest bounding box area) is selected for swapping.
  - If no virtual camera driver is available, the engine runs in preview-only mode and displays actionable guidance in the GUI.

## Testing Decisions

- **Testing Philosophy**: Tests must verify external behavior and invariants, never private internal implementation details. Tests should run deterministically and headlessly without requiring a physical camera device, virtual camera driver, or active display server.
- **Primary Seam**: The entire test suite targets the `FaceSwapEngine` / `Pipeline` boundary:
  - Synthetic and fixture RGB image buffers (with known face orientations, multi-face setups, and zero-face scenarios) are passed into the pipeline.
  - Output assertions verify mathematical consistency: correct bounding box detection, valid 512-dim embedding extraction, stability of the affine transform under the One-Euro filter, and preservation of background pixels during inverse blending.
- **Clean Passthrough Verification**: Test cases verify that passing an image with zero faces or an uninitialized target face returns the exact input pixel buffer untouched.
- **Multi-Face Invariant**: Test cases with multiple synthetic faces verify that the face with the largest area is consistently identified and transformed as the Primary Face.
- **Validation Failure Cases**: Test cases verify that invalid or non-face target images return explicit descriptive errors rather than crashing or returning NaN tensors.
- **Prior Art**: Headless computer vision testing patterns using synthetic image matrices (`image` and `ndarray` fixtures) and pre-recorded test vectors.

## Out of Scope

- Offline batch video file processing (e.g., MP4/MKV file-to-file swapping).
- Multiple simultaneous target face swaps (swapping different identities onto different people in the same frame).
- Heavy neural face restoration (GFPGAN/CodeFormer) in the real-time frame loop.
- Linux and macOS platform-specific virtual camera drivers (focused strictly on Windows 10/11).
- Cloud-based inference or remote video streaming over WebRTC/RTSP.

## Further Notes

- The models required (`inswapper_128.onnx`, `scrfd_2.5g_bnkps.onnx`, and `w600k_r50.onnx`) total approximately 550MB and are cached locally in the user's application data directory.
- DirectML provides hardware acceleration across all DirectX 12 capable GPUs (NVIDIA GeForce, AMD Radeon, Intel Arc/Iris) without external SDK dependencies.
