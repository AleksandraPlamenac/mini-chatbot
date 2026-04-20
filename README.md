---
title: Mini Chatbot
emoji: 🐢
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
license: mit
short_description: Mini chatbot using Hugging Face Transformers
---
## Overview
This project is a small conversational chatbot built with a pretrained Hugging Face transformer model. It responds to user messages in a chat interface without any fine-tuning or backend setup.

The chatbot is implemented in Python using:
- Transformers
- PyTorch
- Gradio
- Hugging Face Spaces

## Goal
The goal is to create a mini chatbot that:
- responds conversationally to user messages
- uses a pretrained model
- is deployed with Gradio on Hugging Face Spaces

## Model Used
- **Model:** `microsoft/DialoGPT-medium`

DialoGPT-medium is a pretrained conversational language model that can generate chatbot-style responses.

## Tech Stack
- **Language model:** microsoft/DialoGPT-medium
- **Interface:** Gradio
- **Hosting:** Hugging Face Spaces
- **Programming language:** Python

## Files Included
- `app.py` — chatbot application code
- `requirements.txt` — required Python packages
- `README.md` — project documentation

## How the Chatbot Works
1. The user types a message.
2. The app keeps a short chat history.
3. The recent conversation is turned into a prompt.
4. DialoGPT generates a response.
5. The response is shown in the Gradio chat interface.

## Features
- conversational chatbot interface
- pretrained transformer model
- short-term conversation memory
- clean Gradio web app
- ready for Hugging Face Spaces deployment

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
