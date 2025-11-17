import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd

class SemanticSearch:
    """Semantic search engine for podcast transcripts."""
    
    def __init__(self, embeddings: np.ndarray, texts: List[str], metadata_df: pd.DataFrame):
        """
        Initialize the search engine.
        
        Args:
            embeddings: Pre-computed text embeddings
            texts: List of text segments corresponding to embeddings
            metadata_df: DataFrame with episode metadata
        """
        self.embeddings = embeddings
        self.texts = texts
        self.metadata_df = metadata_df
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> List[Dict]:
        """
        Search for relevant transcript segments.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of result dictionaries
        """
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by minimum similarity
        top_indices = [idx for idx in top_indices if similarities[idx] >= min_similarity]
        
        # Compile results
        results = []
        for idx in top_indices:
            result = {
                'text': self.texts[idx],
                'similarity': float(similarities[idx]),
                'index': int(idx)
            }
            
            # Add metadata if available
            if idx < len(self.metadata_df):
                row = self.metadata_df.iloc[idx]
                
                if 'episode_title' in row:
                    result['episode_title'] = row['episode_title']
                if 'speaker' in row:
                    result['speaker'] = row['speaker']
                if 'start_time' in row and pd.notna(row['start_time']):
                    result['timestamp'] = self._format_timestamp(row['start_time'])
                if 'episode_id' in row:
                    result['episode_id'] = row['episode_id']
            
            results.append(result)
        
        return results
    
    def search_with_filters(
        self,
        query: str,
        top_k: int = 5,
        episode_filter: str = None,
        speaker_filter: str = None
    ) -> List[Dict]:
        """
        Search with additional filters.
        
        Args:
            query: Search query
            top_k: Number of results
            episode_filter: Filter by episode title
            speaker_filter: Filter by speaker name
            
        Returns:
            Filtered search results
        """
        # Get all results first
        all_results = self.search(query, top_k=len(self.texts))
        
        # Apply filters
        filtered_results = []
        for result in all_results:
            # Episode filter
            if episode_filter and 'episode_title' in result:
                if episode_filter.lower() not in result['episode_title'].lower():
                    continue
            
            # Speaker filter
            if speaker_filter and 'speaker' in result:
                if speaker_filter.lower() not in result['speaker'].lower():
                    continue
            
            filtered_results.append(result)
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
    
    def find_similar_segments(self, segment_index: int, top_k: int = 5) -> List[Dict]:
        """
        Find segments similar to a given segment.
        
        Args:
            segment_index: Index of the reference segment
            top_k: Number of similar segments to return
            
        Returns:
            List of similar segments
        """
        if segment_index >= len(self.embeddings):
            raise ValueError(f"Invalid segment index: {segment_index}")
        
        # Get embedding for this segment
        segment_embedding = self.embeddings[segment_index:segment_index+1]
        
        # Compute similarities
        similarities = cosine_similarity(segment_embedding, self.embeddings)[0]
        
        # Exclude the segment itself
        similarities[segment_index] = -1
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include positive similarities
                result = {
                    'text': self.texts[idx],
                    'similarity': float(similarities[idx]),
                    'index': int(idx)
                }
                
                if idx < len(self.metadata_df):
                    row = self.metadata_df.iloc[idx]
                    if 'episode_title' in row:
                        result['episode_title'] = row['episode_title']
                
                results.append(result)
        
        return results
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in MM:SS format."""
        try:
            seconds = float(seconds)
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        except:
            return str(seconds)
    
    def get_episode_summary_stats(self, episode_title: str) -> Dict:
        """
        Get summary statistics for an episode.
        
        Args:
            episode_title: Title of the episode
            
        Returns:
            Dictionary of statistics
        """
        if 'episode_title' not in self.metadata_df.columns:
            return {}
        
        episode_data = self.metadata_df[self.metadata_df['episode_title'] == episode_title]
        
        if len(episode_data) == 0:
            return {}
        
        stats = {
            'num_segments': len(episode_data),
            'total_words': episode_data['text'].str.split().str.len().sum(),
            'avg_segment_length': episode_data['text'].str.len().mean()
        }
        
        if 'speaker' in episode_data.columns:
            stats['num_speakers'] = episode_data['speaker'].nunique()
            stats['speakers'] = episode_data['speaker'].unique().tolist()
        
        return stats