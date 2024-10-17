import streamlit as st

pages = {
    "Home": [
        st.Page("home.py", title="🎯Home")
    ],
    "Chatbot - beta version": [
        st.Page("chatbot.py", title="🤖AI Chatbot"),
    ],
}

pg = st.navigation(pages)
pg.run()