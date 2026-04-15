from dotenv import load_dotenv
import os
from src.helper import get_load_pdf, get_filter_to_minimal_docs, get_text_split, get_download_huggingface_embedding
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


### starting get in pipieline
extracted_data = get_load_pdf('/data')
filtered_data = get_filter_to_minimal_docs(extracted_data)
text_splitting = get_text_split(filtered_data)
embedding = get_download_huggingface_embedding()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = 'medical-chatbot'

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_splitting,
    index_name=index_name,
    embedding=embedding
)