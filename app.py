import gradio as gr
from transformers import pipeline

chatbot = pipeline("text-generation", model="facebook/blenderbot-400M-distill")

def chatbot_response(message, history):
    result = chatbot(
        message,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7
    )
    return result[0]["generated_text"]

demo = gr.ChatInterface(
    fn=chatbot_response,
    title="Mini Chatbot",
    description="Enter text to start chatting.",
    examples=[
        "Hello",
        "Tell me a joke",
        "What can you do?",
        "Can you help me?",
        "What is a chatbot?"
    ]
)

if __name__ == "__main__":
    demo.launch()