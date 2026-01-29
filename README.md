# C4PS - Captioning, Enhancement, and Multilingual Processing for Socials

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-ff69b4.svg)](https://huggingface.co/docs/transformers/index)
[![Real-ESRGAN](https://img.shields.io/badge/RealESRGAN-Image%20Enhancement-4b8bbe.svg)](https://github.com/xinntao/Real-ESRGAN)
[![Argos Translate](https://img.shields.io/badge/Argos%20Translate-Offline%20Translation-008080.svg)](https://github.com/argosopentech/argos-translate)
[![Discord](https://img.shields.io/badge/Discord-Webhooks-5865F2.svg)](https://discord.com/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

> A modular pipeline for image enhancement, caption generation, and offline multilingual translation. Designed to be used locally (no training required) with optional Discord reporting.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Layout](#project-layout)
- [Available Scripts](#available-scripts)
- [Assets & Weights](#assets--weights)
- [Setup & Installation](#setup--installation)
- [First-Time Setup](#first-time-setup-important)
- [How to Run](#how-to-run-examples)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

C4PS is a local pipeline that performs image enhancement (Real-ESRGAN), caption generation (transformer-based model), and offline translation (Argos Translate). It also includes utilities and scripts for comparisons, weight conversion, and sweep/benchmarking.

This project is designed to be fully local and offline-friendly after the initial model downloads.

---

## Key Features

- Image enhancement using Real-ESRGAN
- Caption generation via a pre-trained transformer model (cached locally)
- Offline multilingual translation with Argos Translate
- Optional Discord webhook reporting
- Terminal UI for interactive runs
- Utilities for model weight handling and comparisons

---

## Tech Stack

- Python 3.12+
- PyTorch
- Hugging Face Transformers (captioning)
- Real-ESRGAN (image upscaling)
- Argos Translate (offline translation)
- NumPy, Pillow

---

## Project Layout

```text
C4PS/
    ├── main.py
    ├── choose_and_run.py
    ├── compare_x4_x4plus.py
    ├── convert_weights.py
    ├── sweep_enhancement.py
    ├── sweep_results.csv
    ├── dummy.pth
    ├── requirements.txt
    ├── README.md
    ├── LICENSE
    ├── assets/
    ├── outputs/
    ├── compare_outputs/
    ├── captioning/
    │   ├── generator.py
    │   └── model.py
    ├── enhancement/
    │   ├── enhancer.py
    │   └── model.py
    ├── translation/
    │   └── translator.py
    ├── utils/
    │   ├── config.py
    │   ├── downloader.py
    │   ├── output_handler.py
    │   └── terminal_ui.py
    ├── gfpgan/
    │   └── weights/
    │       ├── detection_Resnet50_Final.pth
    │       └── parsing_parsenet.pth
    └── weights/
        ├── GFPGANv1.4.pth
        ├── RealESRGAN_x2.pth
        ├── RealESRGAN_x2_chk.pth
        ├── RealESRGAN_x4.pth
        ├── RealESRGAN_x4_chk.pth
        ├── RealESRGAN_x4plus.pth
        ├── transformer_model_cache/
        └── argostranslate/
            └── packages/
```

**Notes:**

- `dummy.pth` is a placeholder weight file included in the repository.
- The transformer model cache and Argos translation packages can be large.

---

## Available Scripts

| Script                 | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| `main.py`              | Full pipeline: enhancement → captioning → translation → output |
| `choose_and_run.py`    | Interactive helper to run parts of the pipeline                |
| `compare_x4_x4plus.py` | Compare ESRGAN x4 vs x4plus outputs                            |
| `convert_weights.py`   | Utility for weight conversion/preprocessing                    |
| `sweep_enhancement.py` | Run enhancement sweeps and generate CSV results                |

---

## Assets & Weights

### Real-ESRGAN & GFPGAN

Stored under `weights/` and `gfpgan/weights/`.

### Captioning Model

Cached automatically under:

```
weights/transformer_model_cache/
```

### Argos Translate (Offline Translation Models)

Argos Translate installs language models into the directory specified by `ARGOS_PACKAGES_DIR`. This repository is configured to store all Argos translation models locally inside the project directory for portability.

---

## Setup & Installation

Example on Windows PowerShell:

```powershell
git clone https://github.com/TheRevanite/C4PS.git
cd C4PS
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

---

## First-Time Setup (IMPORTANT)

Before running the pipeline for the first time, create a `.env` file at the project root.

### Example `.env`

```env
# Required for offline translation
ARGOS_PACKAGES_DIR=weights/argostranslate/packages

# Optional
IMAGE_URL="your_image_url_here"
DISCORD_WEBHOOK_URL="your_discord_webhook_url_here"
DISCORD_SERVER_INVITE="your_discord_server_invite_here"
```

Then run the downloader once:

```powershell
python -m utils.downloader
```

This will:

- Download a sample image (if missing)
- Download Real-ESRGAN weights (if missing)
- Download all configured Argos Translate language models

You only need to do this once per machine.

---

## How to Run (Examples)

Run the full pipeline:

```powershell
python main.py
```

Use the interactive helper:

```powershell
python choose_and_run.py
```

Run enhancement benchmarks:

```powershell
python sweep_enhancement.py
```

Compare ESRGAN variants:

```powershell
python compare_x4_x4plus.py
```

---

## Configuration

Central configuration lives in `utils/config.py`.

### Common Options

| Option                  | Description                                     |
| ----------------------- | ----------------------------------------------- |
| `TARGET_LANGUAGES`      | Languages installed and used by Argos Translate |
| `CAPTIONING_MODEL_PATH` | Captioning model cache path                     |
| `IMAGE_URL`             | Default sample image                            |
| `DISCORD_WEBHOOK_URL`   | Enable Discord reporting if set                 |

Environment variables are loaded automatically via `.env`.

---

## Roadmap

- [x] Image enhancement with Real-ESRGAN
- [x] Transformer-based captioning
- [x] Offline multilingual translation
- [x] Discord webhook integration
- [x] Interactive terminal UI
- [x] Add more language support
- [ ] REST API for external integration
- [ ] Edge device optimization

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Run the pipeline locally
5. Submit a PR

---

## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.
