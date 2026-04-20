import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def build_prompt_from_history(message, history, max_turns=3):
    """
    Building a DialoGPT prompt from recent chat history.
    Only the last few turns are used for keeping generation efficient.
    """
    dialogue = []
    recent_history = history[-max_turns:] if history else []

    for user_msg, bot_msg in recent_history:
        if user_msg:
            dialogue.append(user_msg.strip() + tokenizer.eos_token)
        if bot_msg:
            dialogue.append(bot_msg.strip() + tokenizer.eos_token)

    dialogue.append(message.strip() + tokenizer.eos_token)
    return "".join(dialogue)


def respond(message, history):
    if not message or not message.strip():
        return "Please type a message."

    prompt = build_prompt_from_history(message, history)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.75,
            repetition_penalty=1.2
        )

    response_ids = output_ids[:, input_ids.shape[-1]:]
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()

    if not response:
        response = "I am not sure what to answer. Please try asking something else."

    return response


demo = gr.ChatInterface(
    fn=respond,
    title="Mini Chatbot",
    description="A simple conversational chatbot built with DialoGPT-medium, Transformers, Gradio, and Hugging Face Spaces.",
    examples=[
        "Hello!",
        "What can you do?",
        "Tell me a fun fact about NLP.",
        "Explain chatbots in simple terms."
    ],
    theme=gr.themes.Soft(),
    save_history=True
)

if __name__ == "__main__":
    demo.launch()