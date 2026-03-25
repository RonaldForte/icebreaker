import os
from git import Repo
from langchain_core.documents import Document

def load_github_repo(repo_url, local_path="repo"):

    # Clone repo if not already cloned
    if not os.path.exists(local_path):
        print("Cloning repository...")
        Repo.clone_from(repo_url, local_path)
    else:
        print("Repo already exists locally.")

    documents = []

    # Walk through files
    for root, _, files in os.walk(local_path):
        for file in files:

            # Only load code files (you can expand this later)
            if file.endswith((".py", ".js", ".sql", ".md", ".txt")):

                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": file_path}
                        )
                    )

                except Exception as e:
                    print(f"Skipped {file_path}: {e}")

    return documents