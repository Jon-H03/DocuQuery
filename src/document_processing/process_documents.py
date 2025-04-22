# process_documents.py
import os
import argparse
import logging
import json
from typing import List, Dict, Any

# LangChain imports
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    PyPDFLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_document_loader(file_path: str):
    """Get the appropriate document loader based on file extension."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.txt':
        return TextLoader(file_path)
    elif file_extension in ['.md', '.markdown']:
        return UnstructuredMarkdownLoader(file_path)
    elif file_extension in ['.html', '.htm']:
        return UnstructuredHTMLLoader(file_path)
    elif file_extension == '.pdf':
        return PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def get_text_splitter(file_extension: str, chunk_size: int, chunk_overlap: int):
    """Get the appropriate text splitter based on file extension."""
    if file_extension in ['.md', '.markdown']:
        return RecursiveCharacterTextSplitter.from_language(
            language="markdown",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif file_extension in ['.html', '.htm']:
        return RecursiveCharacterTextSplitter.from_language(
            language="html", 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
    elif file_extension == '.pdf':
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
def save_chunks(chunks: List, output_dir: str, source_filename: str, source_path: str):
    """Save document chunks to disk."""
    # Create subdirectory based on the parent folder of the source file
    parent_dir = os.path.basename(os.path.dirname(source_path))
    subdirectory = os.path.join(output_dir, parent_dir)
    
    # Further organize by document name (without extension)
    doc_name = os.path.splitext(source_filename)[0]
    doc_directory = os.path.join(subdirectory, doc_name)
    
    # Ensure the directory exists
    os.makedirs(doc_directory, exist_ok=True)

    # Save each chunk
    for i, chunk in enumerate(chunks):
        # Create metadata dictionary with additional info
        metadata = chunk.metadata.copy()
        metadata.update({
            "chunk_id": i,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "source_filename": source_filename
        })

        # Create chunk dictionary
        chunk_dict = {
            "content": chunk.page_content,
            "metadata": metadata
        }

        # Create filename
        filename = f"{source_filename.replace('.', '_')}_{i:04d}.json"
        file_path = os.path.join(doc_directory, filename)

        # Save chunk to disk
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(chunks)} chunks to {doc_directory}")

def process_directory(input_dir: str, output_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Process all documents in a directory."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each file in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith('.'):  # Skip hidden files and directories
                continue
        
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file_path)[1].lower()

            try:
                # Get appropraite loader and splitter
                loader = get_document_loader(file_path)
                text_spliiter = get_text_splitter(file_extension, chunk_size, chunk_overlap)

                # Load and split the document
                documents = loader.load()
                chunks = text_spliiter.split_documents(documents)

                # Save chunks
                save_chunks(chunks, output_dir, os.path.basename(file_path), file_path)

                logger.info(f"Processed {file_path} into {len(chunks)} chunks")
            
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

def main():
    # Hardcoded values for development
    input_dir = 'data/raw_docs'
    output_dir = 'data/processed'
    chunk_size = 1000
    chunk_overlap = 200
    
    # Process documents
    process_directory(
        input_dir,
        output_dir,
        chunk_size,
        chunk_overlap
    )

if __name__ == "__main__":
    main()