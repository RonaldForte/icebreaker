import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Cold-Start AI Onboarding", page_icon="🧠")
st.title("Cold-Start AI Onboarding Assistant")

# ---------------------------------------------------
# CRUD PLACEHOLDER
# ---------------------------------------------------
st.header("Items CRUD (Placeholder)")
st.info("These endpoints exist as per project requirements, but we won't use them in the demo.")

if st.button("View Items (GET)"):
    try:
        response = requests.get(f"{API_URL}/items")
        items = response.json()
        if isinstance(items, list) and items:
            for item in items:
                st.write(f"ID: {item.get('id')} | Name: {item.get('name')}")
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
            data = response.json()
            answer = data.get("answer") if isinstance(data, dict) else str(data)
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Enter a question first.")

# ---------------------------------------------------
# VECTOR SEARCH (No LLM)
# ---------------------------------------------------
st.header("Semantic Search (No LLM)")
search_query = st.text_input("Search Code", key="search_query")
collection_name = st.text_input("Collection Name (usually repo name)", key="collection_name")
if st.button("Search"):
    if search_query:
        if not collection_name:
            st.warning("Please enter the collection name for search.")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/vectordb/search",
                    json={"question": search_query, "collection_name": collection_name}
                )
                results = response.json()
                if isinstance(results, list) and results:
                    for doc in results:
                        st.code(doc.get("content", ""), language="python")
                        st.caption(doc.get("metadata", {}).get("source", "Unknown source"))
                else:
                    st.write("No results found.")
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
            data = response.json()
            st.write(data.get("output") if isinstance(data, dict) else str(data))
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Type a message to chat.")