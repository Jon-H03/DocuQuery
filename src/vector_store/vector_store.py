# vector_store.py
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_name="faiss_index", embeddings_model=None):
        self.index_name = index_name
        self.embeddings_model = embeddings_model or OpenAIEmbeddings()  # default to OpenAI model
        self.store = None