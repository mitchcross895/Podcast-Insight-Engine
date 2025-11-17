"""
Embedding Generation Module
Creates semantic embeddings for transcript segments using sentence-transformers
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from tqdm import tqdm
from typing import List, Tuple

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a sentence-transformer model
        all-MiniLM-L6-v2: Fast, efficient, good quality (384 dimensions)
        all-mpnet-base-v2: Better quality but slower (768 dimensions)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        
        # Filter out empty texts
        valid_texts = [t if t else "" for t in texts]
        
        # Generate embeddings with progress bar
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """
        Save embeddings to disk
        """
        np.save(filepath, embeddings)
        print(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """
        Load embeddings from disk
        """
        embeddings = np.load(filepath)
        print(f"Loaded embeddings from {filepath}")
        return embeddings

class VectorSearchIndex:
    def __init__(self, dimension: int):
        """
        Initialize FAISS index for vector similarity search
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.metadata = []
        
    def add_vectors(self, embeddings: np.ndarray, metadata: List[dict]):
        """
        Add vectors to the index with associated metadata
        """
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadata)
        
        print(f"Added {len(embeddings)} vectors to index")
        print(f"Total vectors in index: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[float, dict]]:
        """
        Search for k nearest neighbors
        Returns list of (distance, metadata) tuples
        """
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                results.append((float(dist), self.metadata[idx]))
        
        return results
    
    def save_index(self, index_path: str, metadata_path: str):
        """
        Save FAISS index and metadata
        """
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str):
        """
        Load FAISS index and metadata
        """
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"Loaded index with {self.index.ntotal} vectors")

def create_search_index(df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2') -> Tuple[VectorSearchIndex, EmbeddingGenerator]:
    """
    Create a searchable index from preprocessed transcripts
    """
    # Initialize embedding generator
    embed_gen = EmbeddingGenerator(model_name)
    
    # Generate embeddings
    texts = df['text'].tolist()
    embeddings = embed_gen.generate_embeddings(texts)
    
    # Create metadata for each segment
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            'episode_id': row['episode_id'],
            'episode_title': row['episode_title'],
            'speaker': row['speaker'],
            'text': row['text'],
            'start_time': row.get('start_time', None),
            'end_time': row.get('end_time', None),
            'date': row.get('date', None)
        })
    
    # Create search index
    search_index = VectorSearchIndex(embed_gen.dimension)
    search_index.add_vectors(embeddings, metadata)
    
    return search_index, embed_gen

if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv("preprocessed_transcripts.csv")
    
    # Create search index
    search_index, embed_gen = create_search_index(df)
    
    # Save for later use
    search_index.save_index("faiss_index.idx", "metadata.pkl")
    
    # Test search
    query = "stories about family and relationships"
    query_embedding = embed_gen.model.encode([query])[0]
    results = search_index.search(query_embedding, k=3)
    
    print("\nTest search results:")
    for dist, meta in results:
        print(f"\nDistance: {dist:.4f}")
        print(f"Episode: {meta['episode_title']}")
        print(f"Speaker: {meta['speaker']}")
        print(f"Text: {meta['text'][:200]}...")