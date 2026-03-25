from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # smaller chunks
        chunk_overlap=100    # helps context continuity
    )

    chunked_docs = splitter.split_documents(documents)

    return chunked_docs