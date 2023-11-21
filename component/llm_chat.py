import time
import copy
import openai
import requests
import os
import streamlit as st

from myutils.config import DOCQA_PROMPT, CHAT_EXAMPLES, BAICHUAN_URL, audit_PROMPT
from myutils.es import ElasticsearchServer
from myutils.faiss import FaissDocServer
from myutils.process_data import loadtxt

from .utils import handle_response, create_message

content, labels, backends = loadtxt()
result = {}
for item in content:
    if item[0] in result:
        result[item[0]] += "{}的{}为{}".format(item[0], item[1], item[2])
    else:
        result[item[0]] = "{}的{}为{}".format(item[0], item[1], item[2])
result['其他'] = ''


def llm_base():
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
    st.title("💬代码随行")
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

        st.session_state.messages.append(create_message("user", prompt))
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            tmp = copy.deepcopy(st.session_state.messages)
            for i in reversed(range(len(tmp))):
                if tmp[i]['role'] == 'user':
                    tmp[i][
                        'content'] = f"你是由南京审计大学智能审计团队研发的‘审元’大模型，目前还在不断完善中。\n 如果不是询问身份信息就正常回答。\n <<问题>>： {tmp[i]['content']}"
                    break
            full_response = handle_response(tmp, temperature, history_len,
                                            message_placeholder)
            st.session_state.messages.append(create_message("assistant", full_response))


