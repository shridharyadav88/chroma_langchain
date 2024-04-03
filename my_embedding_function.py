from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from typing import List

class CustomEmbeddings():
    model_name = './models/sentence-transformers/all-MiniLM-L12-v2'

    def embed_documents(self, texts):
        embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
        embeddings = embedding_model.embed_documents(texts)
        return embeddings
    
