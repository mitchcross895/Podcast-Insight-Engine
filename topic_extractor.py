import spacy
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple
import numpy as np

class TopicExtractor:
    """Extracts topics and entities from podcast transcripts."""
    
    def __init__(self):
        """Initialize the topic extractor with NLP models."""
        print("Loading NLP model for entity recognition...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        print("NLP model loaded")
    
    def extract_topics_and_entities(
        self,
        df: pd.DataFrame,
        num_topics: int = 10
    ) -> Tuple[List[Tuple[str, float]], Dict[str, set]]:
        """
        Extract main topics and named entities from the dataset.
        
        Args:
            df: DataFrame with text column
            num_topics: Number of top topics to extract
            
        Returns:
            Tuple of (topics list, entities dict)
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must have 'text' column")
        
        print("Extracting topics...")
        topics = self._extract_topics_tfidf(df, num_topics)
        
        print("Extracting named entities...")
        entities = self._extract_entities(df)
        
        return topics, entities
    
    def _extract_topics_tfidf(self, df: pd.DataFrame, num_topics: int) -> List[Tuple[str, float]]:
        """
        Extract topics using TF-IDF.
        
        Args:
            df: DataFrame with text
            num_topics: Number of topics to return
            
        Returns:
            List of (topic, score) tuples
        """
        # Combine all text
        texts = df['text'].tolist()
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Get feature names and their average scores
            feature_names = vectorizer.get_feature_names_out()
            avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            
            # Sort by score
            top_indices = avg_scores.argsort()[-num_topics:][::-1]
            
            topics = [(feature_names[i], float(avg_scores[i])) for i in top_indices]
            
            return topics
        
        except Exception as e:
            print(f"Error in topic extraction: {e}")
            return []
    
    def _extract_entities(self, df: pd.DataFrame, max_docs: int = 100) -> Dict[str, set]:
        """
        Extract named entities using spaCy.
        
        Args:
            df: DataFrame with text
            max_docs: Maximum number of documents to process (for speed)
            
        Returns:
            Dictionary of entity types to sets of entities
        """
        entities = defaultdict(set)
        
        # Sample documents if dataset is large
        texts = df['text'].tolist()
        if len(texts) > max_docs:
            sample_indices = np.random.choice(len(texts), max_docs, replace=False)
            texts = [texts[i] for i in sample_indices]
        
        # Process texts
        for doc in self.nlp.pipe(texts, batch_size=50):
            for ent in doc.ents:
                # Filter out single-character entities and very common words
                if len(ent.text) > 1 and ent.text.lower() not in ['i', 'we', 'you', 'he', 'she']:
                    entities[ent.label_].add(ent.text)
        
        return dict(entities)
    
    def extract_episode_topics(self, episode_text: str, num_topics: int = 5) -> List[str]:
        """
        Extract topics from a single episode.
        
        Args:
            episode_text: Full episode transcript
            num_topics: Number of topics to extract
            
        Returns:
            List of topic strings
        """
        # Create a temporary DataFrame
        df = pd.DataFrame({'text': [episode_text]})
        
        topics, _ = self.extract_topics_and_entities(df, num_topics)
        
        return [topic for topic, score in topics]
    
    def extract_episode_entities(self, episode_text: str) -> Dict[str, List[str]]:
        """
        Extract entities from a single episode.
        
        Args:
            episode_text: Full episode transcript
            
        Returns:
            Dictionary of entity types to lists of entities
        """
        doc = self.nlp(episode_text[:100000])  # Limit length for processing
        
        entities = defaultdict(list)
        
        for ent in doc.ents:
            if len(ent.text) > 1:
                entities[ent.label_].append(ent.text)
        
        # Convert to regular dict and deduplicate while preserving order
        result = {}
        for label, ent_list in entities.items():
            result[label] = list(dict.fromkeys(ent_list))  # Preserve order, remove duplicates
        
        return result
    
    def tag_episodes_by_topic(self, df: pd.DataFrame, topics: List[str]) -> pd.DataFrame:
        """
        Tag episodes with relevant topics.
        
        Args:
            df: DataFrame with episodes
            topics: List of topic strings to check for
            
        Returns:
            DataFrame with topic tags added
        """
        if 'episode_title' not in df.columns:
            return df
        
        df = df.copy()
        
        # Group by episode
        for episode in df['episode_title'].unique():
            episode_text = ' '.join(
                df[df['episode_title'] == episode]['text'].tolist()
            ).lower()
            
            # Check which topics appear
            episode_topics = [topic for topic in topics if topic.lower() in episode_text]
            
            # Add tags
            df.loc[df['episode_title'] == episode, 'topics'] = ','.join(episode_topics)
        
        return df
    
    def get_topic_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about topics across the dataset.
        
        Args:
            df: DataFrame with text
            
        Returns:
            Dictionary of statistics
        """
        topics, entities = self.extract_topics_and_entities(df, num_topics=20)
        
        stats = {
            'num_topics': len(topics),
            'top_topics': [t[0] for t in topics[:10]],
            'num_entities': sum(len(e) for e in entities.values()),
            'entity_types': list(entities.keys()),
            'num_people': len(entities.get('PERSON', [])),
            'num_places': len(entities.get('GPE', [])),
            'num_organizations': len(entities.get('ORG', []))
        }
        
        return stats