# 🤗 Machine Learning using Hugging Face

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow?style=flat-square)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/Gradio-orange?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](file:///d:/%5B07%5D%20LEARNING++/ml-hugging-face/LICENSE)

Welcome to the Machine Learning using Hugging Face ! This repository is a guide designed to explore the Hugging Face ecosystem, models, and media processing. It covers a wide range of tasks including Natural Language Processing (NLP), Audio Processing, Computer Vision, Video Generation, and web-based Interactive User Interfaces.

---

## 📂 Repository Structure

*   **[transformer_huggingface_tutorial.py](file:///d:/%5B07%5D%20LEARNING++/ml-hugging-face/transformer_huggingface_tutorial.py)**: NLP fundamentals, tokenization, custom text decoding sampling algorithms, sentiment classification, NER, QA, and translation.
*   **[audio_models_hugging_face.py](file:///d:/%5B07%5D%20LEARNING++/ml-hugging-face/audio_models_hugging_face.py)**: Digital Signal Processing (DSP) for audio (STFT, Mel Spectrograms), Audio Spectrogram Transformers (AST), Automatic Speech Recognition (ASR), and Text-to-Speech (TTS).
*   **[diffusers_models_images_hugging_face.py](file:///d:/%5B07%5D%20LEARNING++/ml-hugging-face/diffusers_models_images_hugging_face.py)**: Computer Vision operations, Denoising Diffusion Probabilistic Models (DDPM) for unconditional face generation, and Stable Diffusion XL (SDXL) for text-to-image.
*   **[video_models_hugging_face.py](file:///d:/%5B07%5D%20LEARNING++/ml-hugging-face/video_models_hugging_face.py)**: Generative video modeling using Stable Video Diffusion (SVD) and AnimateDiff pipelines.
*   **[gradio_hugging_face.py](file:///d:/%5B07%5D%20LEARNING++/ml-hugging-face/gradio_hugging_face.py)**: Interactive, web-based UI creation demonstrating layouts, state events, custom styling, and serving models visually.
*   **[requirements.txt](file:///d:/%5B07%5D%20LEARNING++/ml-hugging-face/requirements.txt)**: Python library dependencies list.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed. For running large language models, image generators, and video models, a system with a **CUDA-capable Nvidia GPU** is highly recommended.

### 2. Clone the Repository
```bash
git clone https://github.com/DE-IGNIS/ml-hugging-face.git
cd ml-hugging-face
```

### 3. Create a Virtual Environment
It is highly recommended to run this project inside a virtual environment to avoid dependency conflicts:
```powershell
# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
Install all required libraries specified in [requirements.txt](file:///d:/%5B07%5D%20LEARNING++/ml-hugging-face/requirements.txt):
```bash
pip install -r requirements.txt
```

> [!NOTE]
> If you have a GPU, ensure you install the CUDA-supported version of PyTorch for optimal inference speeds. Check [pytorch.org](https://pytorch.org/) for the command matching your CUDA driver version.

---

## 📖 Module Details

### 📝 1. Natural Language Processing (NLP)
Found in **[transformer_huggingface_tutorial.py](file:///d:/%5B07%5D%20LEARNING++/ml-hugging-face/transformer_huggingface_tutorial.py)**

This module delves into the core mechanics of Transformers:
*   **Tokenization Mechanics**: Demystifies subword tokenization (using GPT-2's vocabulary) on complex words like `pneumonoultramicroscopicsilicovolcanoconiosis` and `homoscedasticity`.
*   **Custom Decoder Sampling Algorithms**:
    *   `[greedy_decode]`: Greedy decoding choosing the maximum probability logits.
    *   `[top_k_sampling]`: Restricts token candidates to the top $K$ items.
    *   `[top_p_sampling]`: Nucleus sampling that filters based on cumulative probability mass $P$.
    *   `[temperature_sampling]`: Distorts logits scale to control output randomness.
*   **High-Level NLP Pipelines**:
    *   *Sentiment Analysis*: Benchmarked on the Stanford IMDB dataset using DistilBERT, and Financial Sentiment Analysis using FinBERT (`ProsusAI/finbert`).
    *   *Named Entity Recognition (NER)*: Automated identification of entities, locations, and organizations.
    *   *Question Answering*: Extraction-based QA using `distilbert-base-cased-distilled-squad`.
    *   *Machine Translation*: Translates text from English to French using `facebook/nllb-200-distilled-600M` via `[translate_sentence]`.

---

### 🔊 2. Audio Processing & Classification
Found in **[audio_models_hugging_face.py]**

Covers DSP operations and audio-native machine learning models:
*   **Digital Signal Processing (DSP)**: Loading audio using `librosa`, plotting waveforms with `matplotlib`, and generating mathematical transforms:
    *   Discrete Fourier Transform (DFT / FFT) with Hanning windowing.
    *   Short-Time Fourier Transform (STFT).
    *   Mel Spectrograms.
*   **Audio Spectrogram Transformer (AST)**: Utilizes MIT's `ast-finetuned-audioset` model to perform deep-learning-based audio classification.
*   **ASR & TTS**:
    *   *Automatic Speech Recognition (ASR)*: Converts speech in audio recordings directly to text using pipelines.
    *   *Text-to-Speech (TTS)*: Synthesizes spoken audio from text strings, visualizes the generated sound waves, and exports the audio as MP3 using `pydub`.

---

### 🎨 3. Image Preprocessing & Generation
Found in **[diffusers_models_images_hugging_face.py]**

Explores classic CV and generative modeling:
*   **Computer Vision Preprocessing**: Channel manipulation (RGB splitting), resizing, and grayscale conversions using `PIL` and `OpenCV`.
*   **Unconditional Generative Modeling**: Implements a Denoising Diffusion Probabilistic Model (DDPM) using `google/ddpm-celebahq-256` to generate realistic human faces. Demonstrates both the high-level `DDPMPipeline` and a manual inference scheduler loop iterating through time steps using `UNet2DModel` on CUDA.
*   **Text-to-Image Generation**: Harnesses Stable Diffusion XL (SDXL) `stabilityai/stable-diffusion-xl-base-1.0` to render ultra-detailed images based on prompt instructions.

---

### 🎬 4. Video Generation & Motion
Found in **[video_models_hugging_face.py]**

Generates motion sequences from static images and prompt-based configurations:
*   **Stable Video Diffusion (SVD)**: Uses `stabilityai/stable-video-diffusion-img2vid-xt` in `float16` precision to transform an image (e.g. `elmo.jpg`) into a high-quality video clip (`elmo.mp4`).
*   **AnimateDiff**: Integrates runwayml's Stable Diffusion v1.5 with the `guoyww/animatediff-motion-adapter-v1-5-2` motion adapter to animate target frames from static images (e.g. `sea.jpg`) combined with descriptive text prompts.
*   **VRAM Optimizations**: Includes memory offload operations to avoid GPU Out-of-Memory (OOM) errors.

---

### 🌐 5. Interactive User Interfaces (Gradio)
Found in **[gradio_hugging_face.py]**

A massive collection of Gradio configurations mapping out interactive dashboards:
*   **Interactive Components**: Number inputs, text inputs, sliders, dropdown lists, image file uploads, and JSON/Label output formats.
*   **Custom Interface Structuring**: Builds intricate dashboard screens using `gr.Blocks()`, layout grids (`gr.Row()`, `gr.Column()`), tab groups (`gr.Tab()`), accordions (`gr.Accordion()`), and buttons.
*   **Complex Event Triggers**: Implements `.change()` and `.click()` event handlers enabling real-time computations (e.g., slider-driven multiplication).
*   **Themes & Styling**: Applies standard visual styling (e.g., `gr.themes.Glass()`) and custom CSS styling injections.
*   **E2E Model Serving**: Builds fully functional web interfaces for:
    *   Image Classification using Microsoft's ResNet-18 model.
    *   Real-time Sentiment Analysis using NLP pipelines.
    *   A hybrid summarization dashboard combining custom NER parsing, Sentiment score tracking, and text generation.

---

## ⚡ Key Optimizations & Best Practices

Running modern generative models locally can be resource-heavy. This playground showcases several essential optimizations to minimize VRAM usage:

1.  **Model CPU Offloading**: Saves GPU memory by offloading components of pipelines to the CPU when not actively processing.
    ```python
    pipe.enable_model_cpu_offload()
    ```
2.  **Attention Slicing**: Breaks up attention computation into smaller sequential steps, reducing memory spikes.
    ```python
    pipe.enable_attention_slicing()
    ```
3.  **Low Precision (Half-Precision)**: Uses `torch.float16` to load models, cutting GPU memory consumption roughly in half.
    ```python
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    ```

---

## 📝 License

This project is licensed under the MIT License 

---

*Made with ❤️ by [DE-IGNIS](https://github.com/DE-IGNIS).*
