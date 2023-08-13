import openai

import gradio as gr
from config import OPENAI_API_KEY, SYSTEM_PROMPT
from openai_helpers import complete, transcribe

openai.api_key = OPENAI_API_KEY

def audio_response(audio, chat_history):
    with open(audio, "rb") as f:
        text = transcribe(f)
    history = [dict(content=content, role=role) for messages in chat_history for role, content in zip(("user", "assistant"), messages)]
    responce, _ = complete(text, history, system=[dict(content=SYSTEM_PROMPT, role="system")])
    chat_history.append((text, responce))
    return None, chat_history


def chat_response(message, chat_history):
    history = [dict(content=content, role=role) for messages in chat_history for role, content in zip(("user", "assistant"), messages)]
    responce, _ = complete(message, history, system=[dict(content=SYSTEM_PROMPT, role="system")])
    chat_history.append((message, responce))
    return "", chat_history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        msg = gr.Textbox()    
        audio_in = gr.Audio(source="microphone", type="filepath")
    audio_in.change(audio_response, [audio_in, chatbot], [audio_in, chatbot])
    msg.submit(chat_response, [msg, chatbot], [msg, chatbot])
    
    clear = gr.ClearButton([msg, chatbot, audio_in])
    
    with gr.Accordion("System Prompt", open=False):
        msg = gr.Textbox(SYSTEM_PROMPT)
        
        def update_prompt(msg):
            global SYSTEM_PROMPT
            SYSTEM_PROMPT = msg
            return SYSTEM_PROMPT

        msg.submit(update_prompt, [msg], [msg])



demo.launch()