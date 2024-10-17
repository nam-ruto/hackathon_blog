import os
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]


def load_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./vector_database_2", embedding_function=embeddings)
    retriever = db.as_retriever()
    print("Load embedding successfully!")
    return retriever


def chat_bot():
    st.title("🤖TroyAI Advisor")
    st.write("👋🏼Hi, how can I help you today?")

    # Display the previous messages
    for message in st.session_state.message_history:
        st.chat_message(message["role"]).write(message["content"])
    
    # Chatbot conversation
    if user_input := st.chat_input("Ask question"):
        st.session_state.message_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Revoke answer
        logging.info("STARTED TO INVOKE")
        conversational_rag_chain = st.session_state.conversational_rag_chain
        response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": "abc123"}}),
        print(response)
        st.session_state.message_history.append({"role": "assistant", "content": response[0]['answer']})
        st.chat_message("assistant").markdown(response[0]['answer'])


def main_logic():
    # Sidebar for API key input
    st.sidebar.subheader("⚙️Settings")
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''

    api_key_input = st.sidebar.text_input("OpenAI API Key:", type="password")

    # Update st.session_state.api_key with the input value if it's provided
    if api_key_input:
        st.session_state.api_key = api_key_input

    # If the API key hasn't been provided, show a message on the main page and stop the app
    if not st.session_state.api_key:
        st.title("🤖TroyAI Advisor")
        st.warning("⚠️ Please input the API key in the sidebar to start using the chatbot service.")
        st.stop()  # Stop further execution until the API key is provided

    try:
        # Load embedding if not already loaded
        if 'retriever' not in st.session_state:
            logging.debug("1. Retriever hasn't been loaded --> Starting to load embeddings")
            st.session_state.retriever = load_embedding()
            logging.debug("2. Load embedding successfully!")

        # Create chat history to display
        if 'message_history' not in st.session_state:
            logging.debug("3. Creating message history")
            st.session_state.message_history = []
            logging.debug("4. Created message history")

        # Create session store
        if 'store' not in st.session_state:
            logging.debug("5. Creating store")
            st.session_state.store = {}
            logging.debug("6. Created store")

        # Create RAG chain if not already created
        if 'conversational_rag_chain' not in st.session_state:
            logging.debug("7. Creating RAG chain")
            
            llm = ChatOpenAI(api_key=st.session_state.api_key, model='gpt-4o-mini')

            # System prompt
            contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            history_aware_retriever = create_history_aware_retriever(
                llm, st.session_state.retriever, contextualize_q_prompt
            )

            system_prompt = (
                """
                You are a friendly Troy University Advisor assistant, please answer the question based only on the following context.
                Context: {context}
                Please answer the question with at least 5 sentences and provide more information from the context.
                Always provide an answer, and never say you are not an expert and not able to answer.
                """
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder('chat_history'),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, prompt)

            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            conversational_rag_chain = RunnableWithMessageHistory(
                                        rag_chain,
                                        get_session_history,
                                        input_messages_key="input",
                                        history_messages_key="chat_history",
                                        output_messages_key="answer",
                                    )

            st.session_state.conversational_rag_chain = conversational_rag_chain

            logging.debug("8. Created RAG chain")

    except Exception as e:
        st.error(f"OpenAI API error: {str(e)}")
        logging.error(f"OpenAI API error: {str(e)}")

    chat_bot()

main_logic()
