from transformers import pipeline, Conversation
import gradio as gr

chatbot = pipeline(model="facebook/blenderbot-400M-distill")

message_list = []
response_list = []

def vanilla_chatbot(message, history):
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

demo_chatbot = gr.ChatInterface(
    vanilla_chatbot,
    title="Vanilla Chatbot",
    description="Enter text to start chatting."
)

demo_chatbot.launch()