import os
from git import Repo
from langchain_core.documents import Document

IGNORE_DIRS = {".git", "node_modules", "venv", "__pycache__", ".idea", ".vscode"} #folders we dont care about
VALID_EXTENSIONS = (".py", ".js", ".sql", ".md", ".txt") # Only the files we want to process through
MAX_FILE_SIZE = 100_000  # 100KB


def load_github_repo(repo_url: str):
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
            if file.endswith(VALID_EXTENSIONS):
                file_path = os.path.join(root, file)

                if os.path.getsize(file_path) > MAX_FILE_SIZE:
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    relative_path = os.path.relpath(file_path, local_path)

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": relative_path,
                                "file_name": file,
                                "file_type": file.split(".")[-1]
                            }
                        )
                    )

                except Exception as e:
                    print(f"Skipped {file_path}: {e}")

    return documents