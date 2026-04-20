import gradio as gr
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

MODEL_NAME = "facebook/blenderbot-400M-distill"

tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)


def build_prompt(message, history):
    prompt = ""

    # history is list of tuples: (user, bot)
    for user_msg, bot_msg in history[-3:]:
        prompt += f"User: {user_msg}\nBot: {bot_msg}\n"

    prompt += f"User: {message}\nBot:"
    return prompt


def chatbot_response(message, history):
    prompt = build_prompt(message, history)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    reply_ids = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7
    )

    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    # Clean output
    if "Bot:" in reply:
        reply = reply.split("Bot:")[-1].strip()

    return reply


demo = gr.ChatInterface(
    fn=chatbot_response,
    title="Mini Chatbot",
    description="Enter text to start chatting."
)


if __name__ == "__main__":
    demo.launch()