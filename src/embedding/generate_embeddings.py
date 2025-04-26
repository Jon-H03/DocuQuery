
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
        # Get relative path from input dir
        rel_path = os.path.relpath(root, input_dir)

        # Create corresponding directory in output
        if rel_path != '.':
            current_output_dir = os.path.join(output_dir, rel_path)
            os.makedirs(current_output_dir, exist_ok=True)
        else:
            current_output_dir = output_dir

        # Process each JSON file
        json_files = [f for f in files if f.endswith('.json')]
        
        if not json_files:
            continue
            
        logger.info(f"Processing {len(json_files)} files in {root}")

        # Batch processing to avoid API limits
        batch_size = 20
        for i in range(0, len(json_files), batch_size):
            batch = json_files[i:i+batch_size]
            texts = []
            file_paths = []

            # Load content from each file
            for filename in batch:
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    texts.append(chunk_data["content"])
                    file_paths.append(file_path)

            # Generate embeddings
            try:
                embeddings = embeddings_model.embed_documents(texts)

                # Save embeddings with original data
                for j, (embedding, file_path) in enumerate(zip(embeddings, file_paths)):
                    # Load original data
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                    
                    # Add embedding to data
                    chunk_data["embedding"] = embedding

                    # Get output file path (maintaining directory structure)
                    rel_file_path = os.path.relpath(file_path, input_dir)
                    output_file_path = os.path.join(output_dir, rel_file_path)

                    # Save updated data
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        json.dump(chunk_data, f)

                logger.info(f"Processed batch {i//batch_size + 1} in {root}")

            except Exception as e:
                logger.error(f"Error processing batch in {root}: {e}")

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