import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Cold-Start AI Onboarding", page_icon="🧠")
st.title("Cold-Start AI Onboarding Assistant")

# ---------------------------------------------------
# Initialize session state
# ---------------------------------------------------
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None

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
            with st.spinner("Loading repo and creating embeddings..."):
                response = requests.post(
                    f"{API_URL}/vectordb/load-repo",
                    json={"repo_url": repo_url},
                    timeout=120  # increase for large repos
                )

            result = response.json()
            message = result.get("message", "Repository loaded!")
            documents_ingested = result.get("documents_ingested", "unknown")

            st.success(f"{message} ({documents_ingested} docs ingested)")

            # Set the active collection
            st.session_state.active_collection = repo_url.split("/")[-1].replace(".git", "")

        except requests.exceptions.Timeout:
            st.error("Loading the repository timed out. Try a smaller repo or increase the timeout.")
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
    if not rag_question:
        st.warning("Enter a question first.")
    elif not st.session_state.active_collection:
        st.warning("No collection selected. Load a repo first.")
    else:
        try:
            with st.spinner("Fetching answer..."):
                response = requests.post(
                    f"{API_URL}/vectordb/rag-query",
                    json={
                        "question": rag_question,
                        "collection_name": st.session_state.active_collection
                    },
                    timeout=60
                )
            data = response.json()
            answer = data.get("answer") if isinstance(data, dict) else str(data)
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------------------------------------
# VECTOR SEARCH (No LLM)
# ---------------------------------------------------
st.header("Semantic Search (No LLM)")
search_query = st.text_input("Search Code", key="search_query")
collection_name_input = st.text_input(
    "Collection Name (usually repo name, optional if you've loaded a repo)",
    key="collection_name"
)

if st.button("Search"):
    if not search_query:
        st.warning("Enter a search query first.")
    else:
        collection_name = collection_name_input or st.session_state.active_collection
        if not collection_name:
            st.warning("No collection selected. Load a repo or enter collection name.")
        else:
            try:
                with st.spinner("Searching..."):
                    response = requests.post(
                        f"{API_URL}/vectordb/search",
                        json={"question": search_query, "collection_name": collection_name},
                        timeout=60
                    )
                results = response.json()
                if isinstance(results, list) and results:
                    for doc in results:
                        st.code(doc.get("content", ""), language="python")
                        st.caption(doc.get("source", "Unknown source"))
                else:
                    st.write("No results found.")
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------------------------------------------
# CHAT WITH MEMORY
# ---------------------------------------------------
st.header("Chat With Memory")
chat_input = st.text_input("Say something", key="chat_input")

if st.button("Send Chat"):
    if not chat_input:
        st.warning("Type a message to chat.")
    else:
        try:
            with st.spinner("Generating response..."):
                response = requests.post(
                    f"{API_URL}/langchain/memory-chat",
                    json={"input": chat_input},
                    timeout=60
                )
            data = response.json()
            st.write(data.get("output") if isinstance(data, dict) else str(data))
        except Exception as e:
            st.error(f"Error: {e}")