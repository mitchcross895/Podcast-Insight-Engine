"""
Summarization Module
Generates episode summaries using local Hugging Face models only
"""

import pandas as pd
from typing import List, Dict
import os
from transformers import pipeline

class EpisodeSummarizer:
    def __init__(self):
        """
        Initialize summarizer with local Hugging Face model
        """
        print("Loading local summarization model...")
        # Using a smaller model for local execution
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  # CPU, use 0 for GPU
        )
        print("Local model loaded successfully")
    
    def summarize_with_local_model(self, text: str, max_length: int = 150) -> str:
        """
        Summarize using local Hugging Face model
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in tokens
            
        Returns:
            Summary string
        """
        # BART works best with texts between 100-1024 tokens
        # Split if too long
        max_input_length = 1024
        words = text.split()
        
        if len(words) > max_input_length:
            # Take first portion for summary
            text = ' '.join(words[:max_input_length])
        
        # Ensure minimum text length
        if len(text.split()) < 50:
            return "Text too short to summarize effectively."
        
        try:
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Summarization error: {e}")
            # Fallback to truncation
            return text[:200] + "..."
    
    def summarize_episode(self, episode_text: str, summary_type: str = "brief") -> str:
        """
        Main summarization method
        
        Args:
            episode_text: Full text to summarize
            summary_type: 'brief' (150 tokens) or 'detailed' (300 tokens)
            
        Returns:
            Summary string
        """
        max_len = 150 if summary_type == "brief" else 300
        return self.summarize_with_local_model(episode_text, max_len)
    
    def batch_summarize_episodes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summaries for all episodes in dataset
        
        Args:
            df: DataFrame with episode data
            
        Returns:
            DataFrame with episode summaries
        """
        print(f"Generating summaries for episodes...")
        
        # Group by episode to get full episode text
        episode_summaries = []
        
        # Get unique episodes
        if 'episode_id' in df.columns:
            episode_ids = df['episode_id'].unique()
        else:
            # Use episode_title if no episode_id
            episode_ids = df['episode_title'].unique() if 'episode_title' in df.columns else range(1)
        
        for episode_id in episode_ids:
            if 'episode_id' in df.columns:
                episode_data = df[df['episode_id'] == episode_id]
            else:
                episode_data = df[df['episode_title'] == episode_id]
            
            if len(episode_data) == 0:
                continue
            
            # Combine all text for episode
            full_text = ' '.join(episode_data['text'].tolist())
            
            # Limit text length for processing
            words = full_text.split()[:2000]  # First ~2000 words
            text_to_summarize = ' '.join(words)
            
            # Generate brief summary
            brief_summary = self.summarize_episode(text_to_summarize, "brief")
            
            # Get episode title
            episode_title = episode_data.iloc[0].get('episode_title', f'Episode {episode_id}')
            
            episode_summaries.append({
                'episode_id': episode_id,
                'episode_title': episode_title,
                'brief_summary': brief_summary,
                'word_count': len(full_text.split()),
                'date': episode_data.iloc[0].get('date', None)
            })
            
            print(f"Summarized: {episode_title}")
        
        return pd.DataFrame(episode_summaries)
    
    def summarize_chunks(self, text: str, chunk_size: int = 1000, summary_type: str = "brief") -> str:
        """
        Summarize very long text by breaking into chunks
        
        Args:
            text: Long text to summarize
            chunk_size: Size of each chunk in words
            summary_type: 'brief' or 'detailed'
            
        Returns:
            Combined summary
        """
        words = text.split()
        
        # If text is short enough, summarize directly
        if len(words) <= 1024:
            return self.summarize_episode(text, summary_type)
        
        # Split into chunks
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  Summarizing chunk {i+1}/{len(chunks)}...")
            summary = self.summarize_episode(chunk, "brief")
            chunk_summaries.append(summary)
        
        # Combine chunk summaries
        combined_summary = ' '.join(chunk_summaries)
        
        # If combined summary is still long, summarize it again
        if len(combined_summary.split()) > 300:
            print("  Creating final summary...")
            return self.summarize_episode(combined_summary, summary_type)
        
        return combined_summary


def generate_highlights(text: str, num_highlights: int = 3) -> List[str]:
    """
    Extract key highlights from transcript
    Simple implementation: extract sentences with emotional words or emphasis
    
    Args:
        text: Text to extract highlights from
        num_highlights: Number of highlights to extract
        
    Returns:
        List of highlight strings
    """
    import re
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Keywords that might indicate important moments
    keywords = ['important', 'realized', 'discovered', 'incredible', 'amazing', 
                'shocking', 'surprised', 'never', 'always', 'remember', 'learned',
                'understand', 'believe', 'moment', 'suddenly', 'finally']
    
    # Score sentences
    scored_sentences = []
    for sent in sentences:
        if len(sent.split()) < 5:  # Skip very short sentences
            continue
        
        score = sum(1 for kw in keywords if kw in sent.lower())
        scored_sentences.append((score, sent.strip()))
    
    # Sort by score and return top highlights
    scored_sentences.sort(reverse=True)
    highlights = [sent for score, sent in scored_sentences[:num_highlights] if sent]
    
    return highlights


def extract_key_quotes(text: str, num_quotes: int = 5) -> List[str]:
    """
    Extract potential key quotes from text
    
    Args:
        text: Text to extract quotes from
        num_quotes: Number of quotes to extract
        
    Returns:
        List of quote strings
    """
    import re
    
    # Find sentences in quotes
    quoted = re.findall(r'"([^"]+)"', text)
    
    if quoted:
        # Return quoted text
        return quoted[:num_quotes]
    
    # Fallback: find sentences with strong language
    sentences = re.split(r'[.!?]+', text)
    
    # Look for sentences with first-person pronouns (often quotes)
    first_person = []
    for sent in sentences:
        if any(word in sent.lower() for word in ['i ', ' i ', "i'm", "i've", "i'd"]):
            if len(sent.split()) > 5:
                first_person.append(sent.strip())
    
    return first_person[:num_quotes]


if __name__ == "__main__":
    # Example usage
    print("Testing EpisodeSummarizer...")
    
    # Check if preprocessed data exists
    import os
    if os.path.exists("data/processed_transcripts.csv"):
        df = pd.read_csv("data/processed_transcripts.csv")
        
        # Initialize summarizer
        summarizer = EpisodeSummarizer()
        
        # Test with first episode
        if 'episode_title' in df.columns:
            first_episode = df['episode_title'].iloc[0]
            episode_data = df[df['episode_title'] == first_episode]
            episode_text = ' '.join(episode_data['text'].tolist())
            
            print(f"\nSummarizing: {first_episode}")
            
            # Brief summary
            brief = summarizer.summarize_episode(episode_text, "brief")
            print(f"\nBrief Summary:\n{brief}")
            
            # Detailed summary
            detailed = summarizer.summarize_episode(episode_text, "detailed")
            print(f"\nDetailed Summary:\n{detailed}")
            
            # Extract highlights
            highlights = generate_highlights(episode_text, 3)
            print(f"\nKey Highlights:")
            for i, highlight in enumerate(highlights, 1):
                print(f"{i}. {highlight}")
        
        # Batch summarize first 5 episodes
        print("\n" + "="*60)
        print("Batch summarizing first 5 episodes...")
        summaries_df = summarizer.batch_summarize_episodes(df.head(100))  # Use head(100) to get ~5 episodes
        
        # Save summaries
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        summaries_df.to_csv(f"{output_dir}/episode_summaries.csv", index=False)
        print(f"\nSummaries saved to {output_dir}/episode_summaries.csv")
        
        # Display sample
        print("\nSample summaries:")
        for i, row in summaries_df.head(3).iterrows():
            print(f"\n{row['episode_title']}:")
            print(f"  {row['brief_summary']}")
    else:
        print("\nNo processed data found. Please run setup_data.py first.")
        print("\nTesting with sample text:")
        
        sample_text = """
        In this episode, we explore the story of a family who moved across the country 
        to start a new life. They faced many challenges along the way, including financial 
        difficulties and cultural adjustments. However, through perseverance and support 
        from their community, they were able to build a successful business. The father 
        says, "I never imagined we would make it this far, but we believed in ourselves 
        and kept pushing forward." Their journey is a testament to the power of determination 
        and the importance of family bonds. Today, they help other immigrant families 
        navigate similar challenges.
        """
        
        summarizer = EpisodeSummarizer()
        summary = summarizer.summarize_episode(sample_text, "brief")
        print(f"\nSummary: {summary}")
        
        highlights = generate_highlights(sample_text, 2)
        print(f"\nHighlights:")
        for h in highlights:
            print(f"  - {h}")