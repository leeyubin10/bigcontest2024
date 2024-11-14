import os
import numpy as np
import faiss
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Define paths
data_path = './data'
module_path = './modules'
csv_file_path = "JEJU_MCT_DATA_modified_v8.csv"
index_path = os.path.join(module_path, 'faiss_index.index')
embedding_array_path = os.path.join(module_path, 'embeddings_array_file.npy')

# Initialize the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
model_name = "intfloat/multilingual-e5-large-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

# Load the dataset
df = pd.read_csv(os.path.join(data_path, csv_file_path))

# Define batch embedding function
def embed_text_batch(text_batch):
    inputs = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# Initialize FAISS index
dimension = 1024  # Update according to your model output dimensions
faiss_index = faiss.IndexFlatL2(dimension)

# Generate embeddings and populate the FAISS index using batch processing
def create_vector_db(dataframe, index, batch_size=16, save=True):
    texts = dataframe['text'].tolist()
    all_embeddings = []

    # Process text in batches
    for i in range(0, len(texts), batch_size):
        text_batch = texts[i:i+batch_size]
        embeddings = embed_text_batch(text_batch)
        all_embeddings.append(embeddings)
        index.add(embeddings)
        print(f"Processed {i + len(text_batch)} texts")

    # Combine all embeddings into a single array
    all_embeddings = np.vstack(all_embeddings)
    print(f"Generated embedding array shape: {all_embeddings.shape}")

    # Save FAISS index and embeddings array if required
    if save:
        faiss.write_index(index, index_path)
        np.save(embedding_array_path, all_embeddings)
        print("FAISS index and embeddings array have been created and saved.")

# Execute the function
if __name__ == "__main__":
    print("Starting the embedding and FAISS index generation process.")
    create_vector_db(df, faiss_index, batch_size=16)  # Adjust batch size as needed
    print("Process completed successfully.")
