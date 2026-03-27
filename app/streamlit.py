import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Cold-Start AI Onboarding", page_icon="🧠")
st.title("Cold-Start AI Onboarding Assistant")

# ---------------------------------------------------
# CRUD PLACEHOLDER
# ---------------------------------------------------
st.header("Items CRUD (Placeholder)") #idk if we need. discuss with Amit
st.info("These endpoints exist as per project requirements, but we won't use them in the demo.")

if st.button("View Items (GET)"):
    try:
        response = requests.get(f"{API_URL}/items")
        items = response.json()
        if items:
            for item in items:
                st.write(f"ID: {item['id']} | Name: {item['name']}")
        else:
            st.write("No items yet.")
    except Exception as e:
        st.error(f"Could not fetch items: {e}")

# ---------------------------------------------------
# REPO LOADER
# ---------------------------------------------------
st.header("Load GitHub Repository")
repo_url = st.text_input("Enter GitHub Repo URL (HTTPS)", key="repo_url")
if st.button("Load Repo"):
    if repo_url:
        try:
            response = requests.post(f"{API_URL}/vectordb/load-repo", json={"repo_url": repo_url})
            st.success(response.json().get("message", "Repository loaded!"))
        except Exception as e:
            st.error(f"Error loading repo: {e}")
    else:
        st.warning("Please enter a GitHub repository URL.")

# ---------------------------------------------------
# RAG QUERY
# ---------------------------------------------------
st.header("Ask About the Codebase")
rag_question = st.text_input("Your Question", key="rag_question")
if st.button("Ask RAG Query"):
    if rag_question:
        try:
            response = requests.post(f"{API_URL}/vectordb/rag-query", json={"question": rag_question})
            st.write(response.json().get("answer", "No response yet."))
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Enter a question first.")

# ---------------------------------------------------
# VECTOR SEARCH (No LLM)
# ---------------------------------------------------
st.header("Semantic Search (No LLM)")
search_query = st.text_input("Search Code", key="search_query")
if st.button("Search"):
    if search_query:
        try:
            response = requests.post(f"{API_URL}/vectordb/search", json={"question": search_query})
            results = response.json()
            for doc in results:
                st.code(doc["content"], language="python")
                st.caption(doc.get("source", "Unknown source"))
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Enter a search query first.")

# ---------------------------------------------------
# CHAT WITH MEMORY
# ---------------------------------------------------
st.header("Chat With Memory")
chat_input = st.text_input("Say something", key="chat_input")
if st.button("Send Chat"):
    if chat_input:
        try:
            response = requests.post(f"{API_URL}/langchain/memory-chat", json={"input": chat_input})
            st.write(response.json())
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Type a message to chat.")