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

## 🔗 Live Demo
👉 https://huggingface.co/spaces/aleksapl/mini-chatbot

## Overview
This project implements a simple conversational chatbot using a pretrained Hugging Face transformer model. The chatbot responds to user input in a web-based chat interface without any fine-tuning or backend setup.

## Goal
The goal of this assignment is to:
- build a chatbot using a pretrained model
- create an interactive interface with Gradio
- deploy the application on Hugging Face Spaces

## Model Used
- **Model:** `facebook/blenderbot-400M-distill`

BlenderBot is a conversational AI model designed for dialogue generation and natural interactions.

## Tech Stack
- **Language model:** facebook/blenderbot-400M-distill  
- **Interface:** Gradio  
- **Hosting:** Hugging Face Spaces  
- **Programming language:** Python  

## Files Included
- `app.py` — chatbot application code  
- `requirements.txt` — required Python packages  
- `README.md` — project documentation  

## How the Chatbot Works
1. The user enters a message in the chat interface.
2. The message is tokenized using the BlenderBot tokenizer.
3. The pretrained model generates a response.
4. The response is decoded into text.
5. The reply is displayed in the Gradio chat interface.

## Features
- conversational chatbot interface  
- pretrained transformer model (no fine-tuning)  
- simple and clean UI using Gradio  
- fully deployed on Hugging Face Spaces  

## Notes
- This implementation uses a pretrained model without additional training.
- Responses may vary due to probabilistic text generation.

---

Check out the configuration reference at:  
https://huggingface.co/docs/hub/spaces-config-reference