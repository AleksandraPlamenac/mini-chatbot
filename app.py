import gradio as gr
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

MODEL_NAME = "facebook/blenderbot-400M-distill"

tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)


def build_prompt(message, history):
    """
    Build conversation context from history.
    Keeps last 3 turns to avoid slowdowns.
    """
    prompt = ""

    # keep last 3 exchanges (user + bot)
    for user_msg, bot_msg in history[-3:]:
        prompt += f"User: {user_msg}\nBot: {bot_msg}\n"

    prompt += f"User: {message}\nBot:"
    return prompt


def chatbot_response(message, history):
    prompt = build_prompt(message, history)

    inputs = tokenizer([prompt], return_tensors="pt")

    reply_ids = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7
    )

    reply = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

    # Clean response (remove prompt echo if it happens)
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