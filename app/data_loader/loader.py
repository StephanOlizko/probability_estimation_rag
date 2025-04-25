from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(texts, metadatas=None, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents(texts = texts, metadatas=metadatas)
    
    return chunks