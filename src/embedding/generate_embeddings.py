
import os
import json
import logging
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_embeddings(input_dir, output_dir, embeddings_model):
    """Generate embeddings for all chunks in a directory and its subdirectories."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all files in directory and its subdirectories
    for root, dirs, files in os.walk(input_dir):
        process_directory(root, files,  input_dir, output_dir, embeddings_model)

def process_directory(root, files, input_dir, output_dir, embeddings_model):
    """Process all JSON files in a single directory."""
    # Get relative path from input dir
    rel_path = os.path.relpath(root, input_dir)
    
    # Create corresponding directory in output
    current_output_dir = create_output_directory(rel_path, output_dir)
    
    # Process each JSON file
    json_files = [f for f in files if f.endswith('.json')]
    
    if not json_files:
        return
        
    logger.info(f"Processing {len(json_files)} files in {root}")
    
    # Process files in batches
    process_files_in_batches(root, json_files, input_dir, output_dir, embeddings_model)

def create_output_directory(rel_path, output_dir):
    """Create corresponding output directory structure."""
    if rel_path != '.':
        current_output_dir = os.path.join(output_dir, rel_path)
        os.makedirs(current_output_dir, exist_ok=True)
        return current_output_dir
    else:
        return output_dir

def process_files_in_batches(root, json_files, input_dir, output_dir, embeddings_model):
    """Process files in batches to avoid API limits."""
    batch_size = 20
    for i in range(0, len(json_files), batch_size):
        batch_files = json_files[i:i+batch_size]
        batch_data = load_batch_data(root, batch_files)
        
        try:
            # Generate embeddings
            embeddings = embeddings_model.embed_documents([data["text"] for data in batch_data])
            
            # Save embeddings
            save_embeddings(batch_data, embeddings, input_dir, output_dir)
            
            logger.info(f"Processed batch {i//batch_size + 1} in {root}")
        
        except Exception as e:
            logger.error(f"Error processing batch in {root}: {e}")

def load_batch_data(root, batch_files):
    """Load content and file paths for a batch of files."""
    batch_data = []
    
    for filename in batch_files:
        file_path = os.path.join(root, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
            batch_data.append({
                "text": chunk_data["content"],
                "file_path": file_path,
                "data": chunk_data
            })
    
    return batch_data

def save_embeddings(batch_data, embeddings, input_dir, output_dir):
    """Save embeddings with original data."""
    for data_item, embedding in zip(batch_data, embeddings):
        # Get original data
        chunk_data = data_item["data"]
        file_path = data_item["file_path"]
        
        # Add embedding to data
        chunk_data["embedding"] = embedding
        
        # Get output file path (maintaining directory structure)
        rel_file_path = os.path.relpath(file_path, input_dir)
        output_file_path = os.path.join(output_dir, rel_file_path)
        
        # Save updated data
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f)

def main():
    # Directories
    processed_dir = 'data/processed'
    embeddings_dir = 'data/embeddings'
    
    # Initialize embedding model
    embeddings_model = OpenAIEmbeddings()
    
    # Generate embeddings for all directories
    generate_embeddings(processed_dir, embeddings_dir, embeddings_model)

if __name__ == "__main__":
    main()