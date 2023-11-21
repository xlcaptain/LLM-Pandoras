import time
import os
import streamlit as st
import pandas as pd

from .pipelines.es import ElasticsearchServer
from .pipelines.utils import handle_response, create_message
from .pipelines.prompt import KNOWLEDGE_PROMPT, CHAT_EXAMPLES

BAICHUAN_URL = os.getenv("BAICHUAN_URL")


def handle_kb_qa(prompt, top_k, threshold):
    index_name = 'audit_index'
    es_server = ElasticsearchServer()
    # es_server.doc_upload(index_name=index_name)
    result = es_server.doc_search(index_name=index_name, query=prompt, top_k=top_k, method='hybrid',
                                  knn_boost=threshold)
    context = "\n".join([doc['content'] for doc in result])
    doc_prompt = KNOWLEDGE_PROMPT.format(query=prompt, context=context)
    reference = [
        {
            "text": doc['content'],
            "source": doc['source'],
            "score": float(doc['score'])
        }
        for doc in result
    ]
    return doc_prompt, reference, True


def knowledge_chat():
    with st.sidebar:
        # TODO: 对话模型与会话绑定
        def on_mode_change():
            st.session_state.messages = []
            mode = st.session_state.vec_modify
            text = f"已切换到 {mode} 模式。"
            if mode == "知识库问答":
                cur_kb = st.session_state.get("selected_kb")
                if cur_kb:
                    text = f"{text} 当前知识库： `{cur_kb}`。"
            st.toast(text)

        # 模型参数选择
        temperature = st.slider("Temperature：", 0.0, 1.0, 0.7, 0.05)
        history_len = st.number_input("历史对话轮数：", 0, 10, 1)

        # 知识库配置

        with st.expander("知识库配置", True):
            vec_modify = st.selectbox("请选择相似度搜索模式：",
                                      ["Elasticsearch",
                                       ],
                                      index=0,
                                      on_change=on_mode_change,
                                      key="vec_modify",
                                      )
            kb_top_k = st.number_input("匹配知识条数：", 1, 6, 5)
            score_threshold = st.slider(
                f"{'知识匹配分数阈值：' if vec_modify == 'Faiss向量库' else '语义关键字权重：(0:代表仅使用关键字)'}：",
                0.0, 1.0, float(0.5), 0.01)

        # 清空对话
        cols = st.columns(2)

        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            st.session_state.messages = []
            st.experimental_rerun()

    st.title("💬 审计知识库问答")
    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter "
    df = pd.DataFrame({"示例": CHAT_EXAMPLES})
    with st.expander("DataFrame", False):
        st.table(df)

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
            with st.spinner("思考中..."):
                doc_prompt, reference, is_true = handle_kb_qa(prompt,
                                                              st.session_state.get("top_k", kb_top_k),
                                                              score_threshold)
            if is_true:
                full_response = handle_response([
                    {"role": st.session_state.messages[-1]["role"], "content": doc_prompt}], 0.1, 1,
                    message_placeholder)
                if reference is not None:
                    st.markdown("### Reference Documents")
                    st.json(reference, expanded=False)
            else:
                for item in doc_prompt:
                    full_response += item
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)
                message_placeholder.markdown(full_response)
                reference = None
            st.session_state.messages.append(create_message("assistant", full_response, reference))
            st.success('Done!')