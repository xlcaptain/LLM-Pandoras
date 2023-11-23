# -*- coding: utf-8 -*-
from component.excel_chat import excel_chat
from component import llm_base, knowledge_chat, case_chat
import streamlit_antd_components as sac
import streamlit as st


def main():
    st.set_page_config(layout="wide")
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    pages = {

        "llm对话": {
            "icon": "chat",
            "func": llm_base,
        },
        "知识库问答": {
            "icon": "chat",
            "func": knowledge_chat,
        },
        "案例问答": {
            "icon": "chat",
            "func": case_chat,
        },
        "表格问答": {
            "icon": "filetype-xlsx",
            "func": excel_chat,
        },
    }

    def on_mode_change():
        st.session_state.messages = []
        mode = st.session_state.selected_page
        text = f"已切换到 {mode} 模式。"
        st.toast(text)

    with st.sidebar:
        st.caption(
            f"""<p align="right">当前版本：1.0.2</p>""",
            unsafe_allow_html=True,
        )

        selected_page = sac.menu([
            sac.MenuItem('对话', icon='box-fill', children=[
                sac.MenuItem('llm对话', icon='chat'),
                sac.MenuItem('知识库问答', icon='book'),
                sac.MenuItem('案例问答', icon='mortarboard-fill'),
                sac.MenuItem('表格问答', icon='filetype-xlsx'),
            ]),
            sac.MenuItem(type='divider'),
        ], format_func='title', open_all=True, index=1, on_change=on_mode_change, key='selected_page')

    if selected_page in pages:
        pages[selected_page]["func"]()


if __name__ == "__main__":
    main()
