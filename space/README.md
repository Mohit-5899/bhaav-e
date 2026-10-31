---
title: Bhaav-E
emoji: 🎞️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
short_description: Text to 2-second animated stickers, 212M model from scratch
models:
  - Mohit5899/bhaav-e
  - Wan-AI/Wan2.2-TI2V-5B-Diffusers
  - google/flan-t5-base
---

# Bhaav-E demo

Type (or build) a prompt and get a 2-second looping sticker, 256 × 256, 8 fps.

- Model: [Mohit5899/bhaav-e](https://huggingface.co/Mohit5899/bhaav-e)
- Code, training and evaluation: [github.com/Mohit-5899/bhaav-e](https://github.com/Mohit-5899/bhaav-e)

Run it with `python app.py` (CPU or GPU; add `--share` for a public link), or on a free GPU with the
[Colab notebook](https://colab.research.google.com/github/Mohit-5899/bhaav-e/blob/main/bhaav_e_demo.ipynb).
The front matter above is ready for hosting as a Gradio Space, which needs a Hugging Face PRO account.
