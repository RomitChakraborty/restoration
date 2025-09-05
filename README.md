# Vintage Cricket Restoration
Colorization + upscaling pipeline for vintage cricket footage on Apple Silicon.  
This repository contains pragmatic, reproducible scripts that combine **DeOldify** (colorization), **RealESRGAN** (video super‑resolution, optional face restoration), and **Gemini** (exemplar still refinement). The code is tuned for **Apple M‑series** (M1/M2+) with **Metal Performance Shaders (MPS)**.

> **Why two Python environments?**  
> In practice, **DeOldify** is most stable for you under **Python 3.10**, while **RealESRGAN/GFPGAN** and your still‑image tools run cleanly under **Python 3.11**. This README keeps those worlds separate so everything “just works.”

---

## Contents
- `colourise_custom.py` — DeOldify wrapper with MPS awareness (video colorization).
- `upscale_video.py` — RealESRGAN video pipeline + ffmpeg muxing (audio preserved), optional **GFPGAN** face enhancement.
- `optimised_upscaler.py` — Still‑image super‑resolution tuned for MPS.
- `model_comparison.py` — Lightweight timing/quality harness for ESRGAN models.
- `gemini_upscale.py`, `gemini_upscale_ref.py` — Exemplar still refinement with **Gemini** (reference‑guided prompts for identity/text fidelity).

---

## System Requirements
- macOS with Apple **M1/M2** (MPS enabled in PyTorch).
- **ffmpeg** installed (`brew install ffmpeg`).
- **Python 3.10** (for DeOldify) and **Python 3.11** (for RealESRGAN/GFPGAN/Gemini).
- Sane disk space (intermediate frames and 4× outputs are large).

---

## Install: two minimal virtualenvs
Create **two** environments at the repo root so you can switch tools without dependency conflicts.

### A) Super‑resolution / Tools env (Python 3.11)
```bash
python3.11 -m venv .venv311 && source .venv311/bin/activate
python -m pip install --upgrade pip
# Core deps (PyTorch includes MPS on Apple Silicon)
pip install torch torchvision torchaudio
# Video SR + face restore + IO
pip install realesrgan gfpgan opencv-python ffmpeg-python tqdm numpy pillow
# Gemini client (new Google library)
pip install google-genai python-dotenv
