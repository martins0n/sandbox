from pathlib import Path
from typing import Literal, Tuple, TypedDict

import openai


class ChatMessage(TypedDict):
    role: Literal["user", "system", "assistant"]
    content: str



def transcribe(audio) -> str:
    transcript = openai.Audio.transcribe("whisper-1", audio)
    text = transcript["text"]
    return text


def complete(text: str, history: list[ChatMessage], openai_kwargs=dict(), system: list[ChatMessage]=[]) -> Tuple[str, list[ChatMessage]]:
    
    history.append({"role": "user", "content": text})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=system + history,
        **openai_kwargs
    )
    message = completion.choices[0].message["content"]
    history.append({"role": "system", "content": message})
    return message, history