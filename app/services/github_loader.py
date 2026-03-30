import os
from git import Repo
from langchain_core.documents import Document

# Ignore generated/dependency folders so the vector store stays focused on user-authored code.
IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".idea", ".vscode"}
MAX_FILE_SIZE = 500_000  # 500KB

# Skip obviously binary/artifact file types to avoid noisy embeddings.
BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".class",
    ".jar",
    ".pyc",
    ".pyd",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
}


def _is_probably_binary(file_path: str) -> bool:
    # A quick null-byte check filters out binary blobs even when file extensions are misleading.
    try:
        with open(file_path, "rb") as f:
            sample = f.read(4096)
        return b"\x00" in sample
    except Exception:
        return True


def load_github_repo(repo_url: str):
    # Repos are cached under ./repos so repeated demos update in place instead of recloning.
    repo_name = repo_url.rstrip("/").split("/")[-1]
    local_path = f"repos/{repo_name}"

    if not os.path.exists(local_path):
        Repo.clone_from(repo_url, local_path)
    else:
        repo = Repo(local_path)
        repo.remotes.origin.pull()

    documents = []

    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            file_path = os.path.join(root, file)
            extension = os.path.splitext(file)[1].lower()

            if extension in BINARY_EXTENSIONS:
                continue

            if os.path.getsize(file_path) > MAX_FILE_SIZE:
                continue

            if _is_probably_binary(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if not content or not content.strip():
                    continue

                relative_path = os.path.relpath(file_path, local_path)

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": relative_path,
                            "file_name": file,
                            "file_type": os.path.splitext(file)[1].lstrip(".").lower(),
                        },
                    )
                )

            except Exception as e:
                print(f"Skipped {file_path}: {e}")

    return documents
