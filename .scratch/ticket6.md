## Parent

Part of #1

## What to build

The complete interactive desktop GUI built with `eframe` / `egui`. Provides a left sidebar featuring a camera selection dropdown, Target Face photo upload button (`rfd`) with thumbnail preview and validation toast, Virtual Camera streaming toggle, and real-time FPS/latency telemetry, alongside a central viewport displaying the live video feed. It connects the physical camera stream, GPU inference worker, and virtual camera sink via bounded lock-free channels.

## Acceptance criteria

- [ ] Renders native desktop window using `eframe` / `egui` with responsive dark theme styling.
- [ ] Left control panel allows selecting from enumerated physical cameras.
- [ ] Includes "Upload Target Face" button launching native OS file dialog (`rfd`) supporting PNG, JPG, and WebP.
- [ ] Displays active Target Face thumbnail in the sidebar with a clear validation error toast when a non-face image is loaded.
- [ ] Central viewport dynamically renders the processed/passthrough video feed as a GPU texture at display refresh rate.
- [ ] Features a Virtual Camera Start/Stop toggle button with live streaming status indicators.
- [ ] Displays an in-app setup guide banner if no virtual camera driver is installed, keeping the GUI preview fully operational.
- [ ] Displays live real-time metrics (FPS counter and pipeline inference latency in ms).
- [ ] Wires camera capture, inference worker, and virtual camera sink across dedicated threads via bounded lock-free channels with zero UI freezing.

## Blocked by

- #4 (Webcam Capture & Zero-Lag Frame Polling)
- #5 (Virtual Camera Output Sink & Driver Fallback)
- #6 (Headless FaceSwapEngine)
