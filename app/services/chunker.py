from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    # Larger chunks preserve enough neighboring code/docs to answer repo-level questions.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=250,
    )
    return splitter.split_documents(documents)
