"""
Summarization Module
Generates episode summaries using LLMs (local or API-based)
"""

import pandas as pd
from typing import List, Dict, Optional
import os
from transformers import pipeline
import anthropic

class EpisodeSummarizer:
    def __init__(self, use_api: bool = False, api_key: Optional[str] = None):
        """
        Initialize summarizer
        use_api: If True, use Claude API; if False, use local Hugging Face model
        """
        self.use_api = use_api
        
        if use_api:
            if not api_key:
                api_key = os.getenv('ANTHROPIC_API_KEY')
            self.client = anthropic.Anthropic(api_key=api_key)
            print("Using Claude API for summarization")
        else:
            print("Loading local summarization model...")
            # Using a smaller model for local execution
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # CPU, use 0 for GPU
            )
            print("Local model loaded")
    
    def summarize_with_local_model(self, text: str, max_length: int = 150) -> str:
        """
        Summarize using local Hugging Face model
        """
        # BART works best with texts between 100-1024 tokens
        # Split if too long
        max_input_length = 1024
        words = text.split()
        
        if len(words) > max_input_length:
            # Take first portion for summary
            text = ' '.join(words[:max_input_length])
        
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
            return text[:200] + "..."  # Fallback to truncation
    
    def summarize_with_api(self, text: str, summary_type: str = "brief") -> str:
        """
        Summarize using Claude API
        """
        if summary_type == "brief":
            prompt = f"""Please provide a brief 2-3 sentence summary of this podcast transcript segment. Focus on the main topic and key points discussed.

Transcript:
{text}

Summary:"""
            max_tokens = 150
        else:  # detailed
            prompt = f"""Please provide a detailed summary of this podcast transcript segment. Include the main themes, key stories or anecdotes, and important points discussed. Organize your summary into clear paragraphs.

Transcript:
{text}

Summary:"""
            max_tokens = 500
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            print(f"API summarization error: {e}")
            return text[:300] + "..."  # Fallback
    
    def summarize_episode(self, episode_text: str, summary_type: str = "brief") -> str:
        """
        Main summarization method
        """
        if self.use_api:
            return self.summarize_with_api(episode_text, summary_type)
        else:
            max_len = 150 if summary_type == "brief" else 300
            return self.summarize_with_local_model(episode_text, max_len)
    
    def batch_summarize_episodes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summaries for all episodes in dataset
        """
        print(f"Generating summaries for {len(df)} episodes...")
        
        # Group by episode to get full episode text
        episode_summaries = []
        
        for episode_id in df['episode_id'].unique():
            episode_data = df[df['episode_id'] == episode_id]
            
            # Combine all text for episode
            full_text = ' '.join(episode_data['text'].tolist())
            
            # Limit text length for processing
            words = full_text.split()[:2000]  # First ~2000 words
            text_to_summarize = ' '.join(words)
            
            # Generate brief and detailed summaries
            brief_summary = self.summarize_episode(text_to_summarize, "brief")
            
            episode_summaries.append({
                'episode_id': episode_id,
                'episode_title': episode_data.iloc[0]['episode_title'],
                'brief_summary': brief_summary,
                'word_count': len(full_text.split()),
                'date': episode_data.iloc[0].get('date', None)
            })
            
            print(f"Summarized: {episode_data.iloc[0]['episode_title']}")
        
        return pd.DataFrame(episode_summaries)

def generate_highlights(text: str, num_highlights: int = 3) -> List[str]:
    """
    Extract key highlights from transcript
    Simple implementation: extract sentences with emotional words or emphasis
    """
    import re
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Keywords that might indicate important moments
    keywords = ['important', 'realized', 'discovered', 'incredible', 'amazing', 
                'shocking', 'surprised', 'never', 'always', 'remember']
    
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

if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv("preprocessed_transcripts.csv")
    
    # Initialize summarizer (use_api=False for local, True for Claude API)
    summarizer = EpisodeSummarizer(use_api=False)
    
    # Generate summaries
    summaries_df = summarizer.batch_summarize_episodes(df.head(10))  # Test with first 10 episodes
    
    # Save summaries
    summaries_df.to_csv("episode_summaries.csv", index=False)
    print("\nSummaries saved to episode_summaries.csv")
    
    # Display sample
    print("\nSample summary:")
    print(summaries_df.iloc[0]['brief_summary'])