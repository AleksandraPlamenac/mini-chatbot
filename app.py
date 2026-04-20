import gradio as gr
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

MODEL_NAME = "facebook/blenderbot-400M-distill"

tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)


def build_prompt(message, history):
    parts = []

    # Keep only recent messages to avoid long/slow prompts
    for item in history[-6:]:
        role = item.get("role", "")
        content = item.get("content", "")

        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Bot: {content}")

    parts.append(f"User: {message}")
    parts.append("Bot:")

    return "\n".join(parts)


def chatbot_response(message, history):
    prompt = build_prompt(message, history)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    reply_ids = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7
    )

    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()

    # Remove prompt echo if BlenderBot repeats it
    if "Bot:" in reply:
        reply = reply.split("Bot:")[-1].strip()

    return reply


demo = gr.ChatInterface(
    fn=chatbot_response,
    title="Mini Chatbot",
    description="Enter text to start chatting.",
    type="messages"
)

if __name__ == "__main__":
    demo.launch()