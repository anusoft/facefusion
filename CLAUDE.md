# FaceFusion — Custom UI + Pipeline

## Goal

Build a custom **UI + processing pipeline** on top of FaceFusion for an easy,
simplified **multi-face → multi-face** ("many-to-many") face swap. The work
lives in a self-contained Gradio layout and reuses FaceFusion's models and
primitives rather than modifying the core engine.

Primary deliverable: `facefusion/uis/layouts/many_to_many.py` (the
**Multi-Face Swap** page).

What the page does today:
- Source faces (multiple photos) → detected and shown as a labelled palette.
- Target **image or video** → faces detected; for video a frame slider picks
  the reference frame used for matching.
- Per-target-face dropdown to choose which source replaces it (explicit match).
- One-click swap. Images swap in-memory; **video** runs one pass where each
  frame's faces are matched to the reference identities by recognition
  embedding, so the right source stays on the right person across the clip.
  Audio is muxed back with ffmpeg.
- **Advanced settings** accordion (collapsed) exposes detector + swapper params.

## Defaults chosen (researched)

- **Swapper:** `inswapper_128` (best identity fidelity) at `256x256` pixel
  boost. "Wide" area option uses `hififace_unofficial_256` (mtcnn_512 crop) to
  cover forehead + jaw — the closest these models get to a whole-head look.
- **Enhancer:** `gfpgan_1.4` post-pass (the biggest quality win; on by default).
- **Detector:** `retinaface` (best for side / profile views; SCRFD and `many`
  are strong alternatives exposed in Advanced).
- **Masks:** `box` + `occlusion` for "Face", `box` only for "Wide".

Hard limitation to remember: FaceFusion swappers replace the **face, not hair /
whole head**. True head/hair swap needs a different model (e.g. GHOST 2.0,
REFace, HeSer) — not integrated here.

## Project layout gotcha

The git repo root is `…/face-fusion-improve/facefusion/` and the Python package
is nested one level deeper at `…/face-fusion-improve/facefusion/facefusion/`.
Layout files go in `facefusion/facefusion/uis/layouts/`. Layouts are
auto-discovered, so adding a file there makes it selectable via
`--ui-layouts <name>`.

## Run it

Dependencies are installed with **uv** into `.venv` (not the official conda
flow):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python facefusion.py run --ui-layouts many_to_many   # serves http://127.0.0.1:7860
```

First launch downloads the full model set (detectors, landmarker, masker,
recognizer, classifier, swapper, plus enhancer/inswapper on first swap).
ffmpeg must be on PATH (used for video audio mux). Apple Silicon → CoreML
execution provider.

## Key engine facts learned

- **`state_manager` has two scopes, `ui` and `cli`**, chosen by
  `app_context.detect_app_context()` which walks the call stack: a frame under
  `facefusion/uis/` → `ui`, under `facefusion/jobs/` → `cli`, else `cli`. So
  `set_item`/`get_item` called from the layout resolve to `ui` state. A test
  script outside those dirs reads `cli` state — this makes model values look
  "unchanged" in standalone prints even though the swap (run through `uis`
  frames) used the value we set. Verify via downloaded model files, not prints.
- All processors' `apply_args` run at startup, so swapper/enhancer/detector
  state items always have defaults regardless of the active `processors` list.
- `read_static_image`/`read_video_frame` return **BGR** (cv2), despite the
  `color_mode='rgb'` default. `write_image` is cv2 → expects BGR. Convert
  BGR→RGB only for Gradio display (`[:, :, ::-1]`).
- `face_swapper` / `face_enhancer` functions live in the package's
  `.core` submodule (the package `__init__.py` is empty). Import
  `…face_swapper.core as face_swapper`.
- `face_swapper.core.swap_face(source_face, target_face, frame)` and
  `face_enhancer.core.enhance_face(face, frame)` are the per-face primitives the
  page calls directly (no full job pipeline needed for the custom flow).
- Reference/identity matching across video frames uses
  `face_selector.calculate_face_distance` on `embedding_norm`.

## Conventions

- Indentation is **tabs** (see `.editorconfig`, `.flake8`).
- flake8 selects `F`, import-order (`I1`/`I2`, pycharm style: plain `import`
  before `from … import`, then alphabetical by module).
- Keep the page self-contained and additive — do not modify the core engine.

## Workflow

- Code is pushed to the fork **`anusoft/facefusion`** (remote `fork`, https →
  rewritten to ssh by a global `insteadOf`). `origin` is upstream
  `facefusion/facefusion`. Commit + push to `fork master` when changes are
  verified.
- Verify changes two ways: a **direct E2E** script (call the layout's functions
  with real example assets) and a **Playwright** browser test against the live
  server. Example assets:
  `examples-3.0.0/source.jpg` and `target-240p.mp4` from facefusion-assets.
