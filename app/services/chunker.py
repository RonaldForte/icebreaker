from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        language=Language.PYTHON, 
        chunk_size=800,      # smaller chunks
        chunk_overlap=100    # helps context continuity
    )

    chunked_docs = splitter.split_documents(documents)

    return chunked_docs