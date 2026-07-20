import os
import gradio as gr

def respond(message, history):
    response = f"You said: {message}\
    \nAnd I said I love learning AI engineering with SDS!"
    return response

port = int(os.environ.get("PORT", 7860))
gr.ChatInterface(fn=respond).launch(server_name="0.0.0.0", server_port=port)