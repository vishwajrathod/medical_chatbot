from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

## Extract data from PDF
def get_load_pdf(file_path):
    loader = DirectoryLoader(
        file_path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


## Minimal documents with page content and source
def get_filter_to_minimal_docs(docs:List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {'source':src}
            )
        )
    return minimal_docs


## Splitting the doument into varioues chunks
def get_text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


## Download the embedding from huggingface
def get_download_huggingface_embedding():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    return embeddings