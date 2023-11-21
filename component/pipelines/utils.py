import openai
import os

BAICHUAN_URL = os.getenv("BAICHUAN_URL")


def create_message(role, content, reference=None):
    message = {"role": role, "content": content}
    if reference is not None:
        message["reference"] = reference
    return message


def handle_response(messages, temperature, history_len, message_placeholder):
    full_response = ""
    openai.api_key = 'xxxx'
    openai.api_base = BAICHUAN_URL
    for response in openai.ChatCompletion.create(
            model="baichuan",
            messages=messages[-history_len * 2 - 1:],
            temperature=temperature,
            stream=True,
    ):
        full_response += response.choices[0].delta.get("content", "")
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    return full_response
