import os
import requests
import streamlit as st


def detect_api_url() -> str:
    # Streamlit defaults to the local FastAPI server and verifies it before proceeding.
    preferred = os.getenv("API_URL", "http://127.0.0.1:8000").strip()
    if not preferred:
        preferred = "http://127.0.0.1:8000"

    try:
        response = requests.get(f"{preferred}/openapi.json", timeout=3)
        if response.status_code == 200:
            return preferred
    except Exception:
        pass

    return "http://127.0.0.1:8000"


def get_error_message(response: requests.Response) -> str:
    # Normalize backend errors so the UI shows the real FastAPI/HTTP detail message.
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return str(payload.get("detail") or payload.get("error") or payload)
        return str(payload)
    except Exception:
        return response.text or f"HTTP {response.status_code}"


def api_request(method: str, path: str, payload: dict | None = None, timeout: int = 30):
    response = requests.request(
        method=method,
        url=f"{st.session_state.api_url}{path}",
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        return False, get_error_message(response)

    try:
        return True, response.json()
    except Exception:
        return True, {}


def fetch_openapi_paths(api_url: str, timeout: int = 5) -> set[str]:
    # Streamlit uses the OpenAPI document to detect stale backend processes early.
    try:
        response = requests.get(f"{api_url}/openapi.json", timeout=timeout)
        if response.status_code >= 400:
            return set()
        payload = response.json()
        paths = payload.get("paths", {}) if isinstance(payload, dict) else {}
        return set(paths.keys()) if isinstance(paths, dict) else set()
    except Exception:
        return set()


def refresh_users():
    # Users are stored server-side in memory, so the UI refreshes them on demand.
    ok, data = api_request("GET", "/users", timeout=10)
    if not ok:
        return

    if isinstance(data, list):
        st.session_state.users = data
        if data and st.session_state.selected_user_id is None:
            st.session_state.selected_user_id = data[0].get("id")


# The app is intentionally linear: choose a user, load a repo, explore it, then chat.
st.set_page_config(page_title="Cold-Start AI Onboarding", page_icon="🧠")
st.title("Cold-Start AI Onboarding Assistant")

if "api_url" not in st.session_state:
    st.session_state.api_url = detect_api_url()
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
if "users" not in st.session_state:
    st.session_state.users = []
if "selected_user_id" not in st.session_state:
    st.session_state.selected_user_id = None
if "api_paths" not in st.session_state:
    st.session_state.api_paths = set()

api_paths = fetch_openapi_paths(st.session_state.api_url, timeout=5)
st.session_state.api_paths = api_paths

if api_paths:
    st.success(f"Connected to API: {st.session_state.api_url}")
    if "/vectordb/code-summary" not in api_paths:
        st.warning(
            "Connected backend is outdated (missing /vectordb/code-summary). "
            "Stop old uvicorn terminals and restart one clean server."
        )
else:
    st.error(
        f"Could not connect to API at {st.session_state.api_url}. Start FastAPI and refresh this page."
    )

if not st.session_state.users:
    refresh_users()

st.subheader("Step 1: Select User")
user_col1, user_col2 = st.columns([2, 1])

with user_col1:
    new_username = st.text_input("Create User", placeholder="e.g. ronal")
with user_col2:
    if st.button("Add User"):
        if not new_username.strip():
            st.warning("Enter a username first.")
        else:
            user_id = len(st.session_state.users) + 1
            ok, data = api_request(
                "POST",
                "/users",
                payload={"id": user_id, "username": new_username.strip()},
                timeout=10,
            )
            if not ok:
                st.error(data)
            elif isinstance(data, dict) and data.get("error"):
                st.warning(data["error"])
            else:
                refresh_users()
                st.session_state.selected_user_id = user_id
                st.success("User created.")

if st.session_state.users:
    options = {
        f"{u.get('username')} (ID: {u.get('id')})": u.get("id")
        for u in st.session_state.users
    }
    labels = list(options.keys())

    idx = 0
    if st.session_state.selected_user_id is not None:
        for i, label in enumerate(labels):
            if options[label] == st.session_state.selected_user_id:
                idx = i
                break

    active_label = st.selectbox("Active User", labels, index=idx)
    st.session_state.selected_user_id = options[active_label]
else:
    st.info("Create a user to continue.")

st.subheader("Step 2: Load Repository")
repo_url = st.text_input("GitHub Repo URL", placeholder="https://github.com/org/repo")

if st.button("Load Repo"):
    if not repo_url.strip():
        st.warning("Enter a GitHub repository URL.")
    else:
        with st.spinner("Loading repository and creating embeddings..."):
            ok, data = api_request(
                "POST",
                "/vectordb/load-repo",
                payload={"repo_url": repo_url.strip()},
                timeout=180,
            )
        if not ok:
            st.error(data)
        else:
            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
            st.session_state.active_collection = repo_name
            st.success(
                f"{data.get('message', 'Repository loaded.')} ({data.get('documents_ingested', 'unknown')} docs)"
            )

if st.session_state.active_collection:
    st.caption(f"Active Repository Collection: {st.session_state.active_collection}")

st.subheader("Step 3: Explore Repository")
if not st.session_state.active_collection:
    st.info("Load a repo first to enable summary/search/Q&A.")
else:
    summary_prompt = st.text_input(
        "Grounded Summary Prompt",
        value="Summarize what this project does and the main components.",
    )
    if st.button("Generate Grounded Summary"):
        if "/vectordb/code-summary" not in st.session_state.api_paths:
            st.error(
                "This backend does not expose /vectordb/code-summary. "
                "Restart uvicorn from this workspace and try again."
            )
        else:
            ok, data = api_request(
                "POST",
                "/vectordb/code-summary",
                payload={
                    "prompt": summary_prompt,
                    "collection_name": st.session_state.active_collection,
                    "k": 8,
                },
                timeout=90,
            )
            if not ok:
                st.error(data)
            else:
                st.write(data.get("summary", "No summary returned."))

    search_query = st.text_input(
        "Semantic Search Query", placeholder="e.g. authentication"
    )
    if st.button("Run Semantic Search"):
        if not search_query.strip():
            st.warning("Enter a search query first.")
        else:
            ok, data = api_request(
                "POST",
                "/vectordb/search",
                payload={
                    "question": search_query.strip(),
                    "collection_name": st.session_state.active_collection,
                },
                timeout=60,
            )
            if not ok:
                st.error(data)
            elif isinstance(data, list) and data:
                # Raw search results stay visible so users can inspect exactly what was retrieved.
                for doc in data:
                    st.code(doc.get("content", ""), language="python")
                    st.caption(doc.get("source", "Unknown source"))
            else:
                st.info("No results found.")

    rag_question = st.text_input("Ask a Repository Question")
    if st.button("Ask Repo Q&A"):
        if not rag_question.strip():
            st.warning("Enter a question first.")
        else:
            ok, data = api_request(
                "POST",
                "/vectordb/rag-query",
                payload={
                    "question": rag_question.strip(),
                    "collection_name": st.session_state.active_collection,
                },
                timeout=60,
            )
            if not ok:
                st.error(data)
            else:
                st.write(data.get("answer", "No answer returned."))

st.subheader("Step 4: Chat With Memory")
chat_col1, chat_col2 = st.columns([3, 1])

with chat_col1:
    chat_input = st.text_input("Message")
with chat_col2:
    reset_clicked = st.button("Reset Memory")

if reset_clicked:
    if st.session_state.selected_user_id is None:
        st.warning("Create/select a user first.")
    else:
        ok, data = api_request(
            "POST",
            f"/langchain/memory-chat/reset/{int(st.session_state.selected_user_id)}",
            timeout=20,
        )
        if not ok:
            st.error(data)
        else:
            st.success(data.get("message", "Memory reset."))

if st.button("Send Chat"):
    if not chat_input.strip():
        st.warning("Type a message to chat.")
    elif st.session_state.selected_user_id is None:
        st.warning("Create/select a user first.")
    else:
        payload = {
            "user_id": int(st.session_state.selected_user_id),
            "input": chat_input.strip(),
        }

        # When a repo is active, the backend blends repository grounding with user memory.
        if st.session_state.active_collection:
            payload["collection_name"] = st.session_state.active_collection

        ok, data = api_request(
            "POST",
            "/langchain/memory-chat",
            payload=payload,
            timeout=60,
        )
        if not ok:
            st.error(data)
        elif isinstance(data, dict):
            st.write(
                data.get("output") or data.get("response") or "No response returned."
            )
            sources = data.get("sources", [])
            if sources:
                st.caption("Sources: " + ", ".join(dict.fromkeys(sources)))
        else:
            st.write(str(data))
