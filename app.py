import gradio as gr
from transformers import pipeline, Conversation

chatbot = pipeline("conversational", model="facebook/blenderbot-400M-distill")

message_list = []
response_list = []

def chatbot_response(message, history):
    conversation = Conversation(
        text=message,
        past_user_inputs=message_list,
        generated_responses=response_list
    )

    conversation = chatbot(conversation)
    reply = conversation.generated_responses[-1]

    message_list.append(message)
    response_list.append(reply)

    return reply

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