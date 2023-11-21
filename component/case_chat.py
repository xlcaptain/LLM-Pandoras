import time
import copy
import requests
import streamlit as st
from .pipelines.process_data import loadtxt
from .pipelines.utils import handle_response, create_message
from .pipelines.prompt import CASE_PROMPT


content, labels, backends = loadtxt()
result = {}
for item in content:
    if item[0] in result:
        result[item[0]] += "{}的{}为{}".format(item[0], item[1], item[2])
    else:
        result[item[0]] = "{}的{}为{}".format(item[0], item[1], item[2])
result['其他'] = ''


def case_chat():

    with st.sidebar:
        # 模型参数选择
        temperature = st.slider("Temperature：", 0.0, 1.0, 0.7, 0.05)
        history_len = st.number_input("历史对话轮数：", 0, 10, 1)

        # 清空对话
        cols = st.columns(2)

        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            st.session_state.messages = []
            st.experimental_rerun()

    # Display chat messages from history on app rerun
    st.title("💬审计案例问答")
    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter "

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        full_response = ''
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = requests.post("http://192.168.20.59:9001/predict", json={'text': prompt})

            response = next(iter(response.json()))
            full_response = ''

            if response and response != '其他':
                tmp = copy.deepcopy(st.session_state.messages)
                for i in reversed(range(len(tmp))):
                    if tmp[i]['role'] == 'user':
                        tmp[i]['content'] = CASE_PROMPT.format(query=prompt, context=result[response])
                        break
                full_response = handle_response(tmp, temperature, history_len,
                                                message_placeholder)
                st.markdown("### Reference Documents")
                st.json({'text': result[response], 'type': response}, expanded=False)

            else:
                for response in list('对不起，我无法识别你的案例类型，所以无法回答你的问题，请重新输入。'):
                    full_response += response
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            st.session_state.messages.append(create_message("assistant", full_response))