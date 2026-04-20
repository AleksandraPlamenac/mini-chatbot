import gradio as gr
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

MODEL_NAME = "facebook/blenderbot-400M-distill"

tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)


def chatbot_response(message, history):
    inputs = tokenizer(message, return_tensors="pt")

    reply_ids = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7
    )

    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply


demo = gr.ChatInterface(
    fn=chatbot_response,
    title="Mini Chatbot",
    description="Enter text to start chatting."
)

if __name__ == "__main__":
    demo.launch()