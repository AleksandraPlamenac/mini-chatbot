---
title: Mini Chatbot
emoji: 🤖
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
license: mit
short_description: Mini chatbot using BlenderBot and Hugging Face Transformers
---

## Overview
This project is a simple conversational chatbot built using a pretrained Hugging Face transformer model. It allows users to interact with the model through a clean web interface without any fine-tuning or backend setup.

The chatbot is deployed using Gradio on Hugging Face Spaces.

---

## Goal
The goal of this assignment is to:
- build a chatbot using a pretrained transformer model
- create an interactive interface with Gradio
- deploy the application on Hugging Face Spaces

---

## Model Used
- **Model:** `facebook/blenderbot-400M-distill`

BlenderBot is a conversational model developed by Facebook AI that is designed for open-domain dialogue generation.

---

## Tech Stack
- **Language model:** BlenderBot (Hugging Face Transformers)
- **Interface:** Gradio
- **Hosting:** Hugging Face Spaces
- **Programming language:** Python

---

## Files Included
- `app.py` — chatbot application code  
- `requirements.txt` — Python dependencies  
- `README.md` — project documentation  

---

## How the Chatbot Works
1. The user enters a message in the chat interface.
2. The message is tokenized using the BlenderBot tokenizer.
3. The model generates a response based on the input.
4. The response is decoded and displayed in the interface.

---

## Features
- conversational chatbot interface  
- pretrained transformer model  
- simple and responsive UI using Gradio  
- deployed on Hugging Face Spaces  

---

## Notes on Conversation Memory
Conversation memory was tested as an optional improvement. However, due to limitations of the BlenderBot model in this environment (CPU-based Hugging Face Spaces), multi-turn conversation history caused instability and runtime errors.

To ensure a stable and responsive chatbot, the final version processes each message independently without maintaining long conversation history.

---

## Future Improvements
- implement stable conversation memory using a different model (e.g., DialoGPT)  
- enhance UI design and styling  
- experiment with larger or more advanced conversational models  
- add user controls for response length and creativity  

---

## Live Demo
👉 https://huggingface.co/spaces/aleksapl/mini-chatbot

---

Check out the configuration reference:  
https://huggingface.co/docs/hub/spaces-config-reference