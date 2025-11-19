#!/usr/bin/env python3
"""
Podcast Insight Engine - Command Line Interface
Main CLI application replacing the Streamlit interface
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint
import cmd

# Import core modules
from data_processor import DataProcessor
from embedding_generator import EmbeddingGenerator
from semantic_search import SemanticSearch
from summarizer import EpisodeSummarizer
from topic_extractor import TopicExtractor
from config import *

console = Console()

class PodcastCLI:
    """Main CLI application class"""
    
    def __init__(self):
        """Initialize CLI with configuration"""
        self.console = Console()
        self.data_loaded = False
        self.embeddings_generated = False
        self.processed_data = None
        self.embeddings = None
        self.embedding_texts = None
        self.searcher = None
        
        # Try to load cached data
        self._load_cache()
    
    def _load_cache(self):
        """Load cached data if available"""
        try:
            cache_file = Path('cache/embeddings.npy')
            metadata_file = Path('cache/metadata.pkl')
            
            if cache_file.exists() and metadata_file.exists():
                self.embeddings = np.load(cache_file)
                import pickle
                with open(metadata_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.embedding_texts = cache_data.get('texts', [])
                    self.processed_data = cache_data.get('dataframe')
                
                if self.processed_data is not None:
                    self.data_loaded = True
                    self.embeddings_generated = True
                    self.searcher = SemanticSearch(
                        self.embeddings,
                        self.embedding_texts,
                        self.processed_data
                    )
                    self.console.print("[dim]Loaded cached data and embeddings[/dim]")
        except Exception as e:
            pass  # Silent fail, will load data when needed
    
    def _save_cache(self):
        """Save data to cache"""
        try:
            cache_dir = Path('cache')
            cache_dir.mkdir(exist_ok=True)
            
            if self.embeddings is not None:
                np.save('cache/embeddings.npy', self.embeddings)
            
            if self.embedding_texts is not None and self.processed_data is not None:
                import pickle
                with open('cache/metadata.pkl', 'wb') as f:
                    pickle.dump({
                        'texts': self.embedding_texts,
                        'dataframe': self.processed_data
                    }, f)
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not save cache: {e}[/yellow]")
    
    def load_data(self, filepath):
        """Load and process data from CSV"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Loading data...", total=None)
            
            try:
                df = pd.read_csv(filepath)
                processor = DataProcessor()
                self.processed_data = processor.process_dataframe(df)
                self.data_loaded = True
                
                progress.update(task, completed=True)
                self.console.print(f"[green]✓[/green] Loaded {len(self.processed_data)} transcript segments")
                
                return True
            except Exception as e:
                self.console.print(f"[red]✗[/red] Error loading data: {e}")
                return False
    
    def generate_embeddings(self, batch_size=32):
        """Generate embeddings for loaded data"""
        if not self.data_loaded:
            self.console.print("[red]✗[/red] No data loaded. Load data first.")
            return False
        
        self.console.print("[cyan]Generating embeddings...[/cyan]")
        self.console.print("This may take a while depending on dataset size.")
        
        try:
            embed_gen = EmbeddingGenerator()
            
            # Use the updated method signature
            self.embeddings, self.embedding_texts = embed_gen.generate_embeddings(
                self.processed_data, 
                batch_size=batch_size
            )
            self.embeddings_generated = True
            
            # Initialize searcher
            self.searcher = SemanticSearch(
                self.embeddings,
                self.embedding_texts,
                self.processed_data
            )
            
            # Save to cache
            self._save_cache()
            
            self.console.print(f"[green]✓[/green] Generated {len(self.embeddings)} embeddings")
            return True
            
        except Exception as e:
            self.console.print(f"[red]✗[/red] Error generating embeddings: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cmd_search(self, query, top_k=5, min_similarity=0.0, output=None):
        """Execute search command"""
        if not self.embeddings_generated:
            self.console.print("[red]✗[/red] Embeddings not generated. Run 'embed' command first.")
            return None
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Searching...", total=None)
            
            try:
                results = self.searcher.search(query, top_k=top_k, min_similarity=min_similarity)
                progress.update(task, completed=True)
                
                # Display results
                self._display_search_results(results, query)
                
                # Save to file if requested
                if output:
                    self._save_results(results, output)
                
                return results
                
            except Exception as e:
                self.console.print(f"[red]✗[/red] Search error: {e}")
                return None
    
    def _display_search_results(self, results, query):
        """Display search results in formatted table"""
        if not results:
            self.console.print(f"[yellow]No results found for: '{query}'[/yellow]")
            return
        
        self.console.print(f"\n[bold cyan]Search Results for:[/bold cyan] '{query}'")
        self.console.print(f"[dim]Found {len(results)} results[/dim]\n")
        
        for idx, result in enumerate(results, 1):
            # Create panel for each result
            similarity = result['similarity']
            color = "green" if similarity > 0.7 else "yellow" if similarity > 0.5 else "white"
            
            title = f"Result {idx} - Similarity: [{color}]{similarity:.3f}[/{color}]"
            
            content = []
            if 'episode_title' in result:
                content.append(f"[bold]Episode:[/bold] {result['episode_title']}")
            if 'speaker' in result and result['speaker']:
                content.append(f"[bold]Speaker:[/bold] {result['speaker']}")
            if 'timestamp' in result and result['timestamp']:
                content.append(f"[bold]Time:[/bold] {result['timestamp']}")
            
            # Truncate text if too long
            text = result['text']
            if len(text) > 400:
                text = text[:400] + "..."
            content.append(f"\n{text}")
            
            panel = Panel(
                "\n".join(content),
                title=title,
                border_style=color
            )
            self.console.print(panel)
            self.console.print()
    
    def cmd_summarize(self, episode, summary_type="brief", use_api=False):
        """Execute summarize command"""
        if not self.data_loaded:
            self.console.print("[red]✗[/red] No data loaded.")
            return None
        
        # Get episode text
        episode_data = self.processed_data[
            self.processed_data['episode_title'] == episode
        ]
        
        if len(episode_data) == 0:
            self.console.print(f"[red]✗[/red] Episode not found: '{episode}'")
            # Show available episodes
            if 'episode_title' in self.processed_data.columns:
                episodes = self.processed_data['episode_title'].unique()[:10]
                self.console.print("\n[dim]Available episodes (showing first 10):[/dim]")
                for ep in episodes:
                    self.console.print(f"  - {ep}")
            return None
        
        episode_text = " ".join(episode_data['text'].tolist())
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task(f"Generating {summary_type} summary...", total=None)
            
            try:
                summarizer = EpisodeSummarizer(use_api=use_api)
                summary = summarizer.summarize_episode(episode_text, summary_type)
                progress.update(task, completed=True)
                
                # Display summary
                self.console.print()
                panel = Panel(
                    summary,
                    title=f"[bold cyan]Summary: {episode}[/bold cyan]",
                    border_style="cyan"
                )
                self.console.print(panel)
                
                return summary
                
            except Exception as e:
                self.console.print(f"[red]✗[/red] Summarization error: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    def cmd_topics(self, num_topics=10, episode=None, output=None):
        """Execute topics command"""
        if not self.data_loaded:
            self.console.print("[red]✗[/red] No data loaded.")
            return None
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Extracting topics...", total=None)
            
            try:
                extractor = TopicExtractor()
                
                # Filter by episode if specified
                data = self.processed_data
                if episode:
                    data = data[data['episode_title'] == episode]
                    if len(data) == 0:
                        self.console.print(f"[red]✗[/red] Episode not found: '{episode}'")
                        return None
                
                topics, entities = extractor.extract_topics_and_entities(data, num_topics)
                progress.update(task, completed=True)
                
                # Display topics
                self._display_topics(topics, entities, episode)
                
                # Save if requested
                if output:
                    self._save_topics(topics, entities, output)
                
                return topics, entities
                
            except Exception as e:
                self.console.print(f"[red]✗[/red] Topic extraction error: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    def _display_topics(self, topics, entities, episode=None):
        """Display topics and entities"""
        title = "Extracted Topics"
        if episode:
            title += f" - {episode}"
        
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]\n")
        
        # Topics table
        table = Table(title="Topics", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Topic", style="cyan")
        table.add_column("Score", justify="right", style="green")
        
        for idx, (topic, score) in enumerate(topics, 1):
            table.add_row(str(idx), topic, f"{score:.3f}")
        
        self.console.print(table)
        self.console.print()
        
        # Entities
        if entities:
            self.console.print("[bold cyan]Named Entities[/bold cyan]\n")
            
            for entity_type, entity_set in entities.items():
                if entity_set:
                    entities_list = list(entity_set)[:15]  # Show first 15
                    self.console.print(f"[bold]{entity_type}:[/bold]")
                    for entity in entities_list:
                        self.console.print(f"  • {entity}")
                    self.console.print()
    
    def cmd_stats(self, episode=None, detailed=False):
        """Execute stats command"""
        if not self.data_loaded:
            self.console.print("[red]✗[/red] No data loaded.")
            return
        
        data = self.processed_data
        
        if episode:
            data = data[data['episode_title'] == episode]
            if len(data) == 0:
                self.console.print(f"[red]✗[/red] Episode not found: '{episode}'")
                return
        
        # Calculate statistics
        stats = {
            'Total Segments': len(data),
            'Total Words': data['text'].str.split().str.len().sum(),
            'Avg Segment Length': f"{data['text'].str.len().mean():.0f} chars",
        }
        
        if 'episode_title' in data.columns:
            stats['Total Episodes'] = data['episode_title'].nunique()
        
        if 'speaker' in data.columns:
            stats['Total Speakers'] = data['speaker'].nunique()
        
        # Display
        title = "Dataset Statistics"
        if episode:
            title = f"Statistics - {episode}"
        
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]\n")
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="green")
        
        for metric, value in stats.items():
            table.add_row(metric, str(value))
        
        self.console.print(table)
        self.console.print()
        
        if detailed and 'episode_title' in data.columns:
            # Episode breakdown
            self.console.print("[bold cyan]Top 10 Episodes by Segment Count[/bold cyan]\n")
            
            episode_counts = data['episode_title'].value_counts().head(10)
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Rank", style="dim", width=6)
            table.add_column("Episode", style="cyan")
            table.add_column("Segments", justify="right", style="green")
            
            for idx, (ep, count) in enumerate(episode_counts.items(), 1):
                table.add_row(str(idx), ep[:60], str(count))
            
            self.console.print(table)
    
    def cmd_list_episodes(self, limit=20):
        """List available episodes"""
        if not self.data_loaded:
            self.console.print("[red]✗[/red] No data loaded.")
            return
        
        if 'episode_title' in self.processed_data.columns:
            episodes = self.processed_data['episode_title'].unique()
            
            self.console.print(f"\n[bold cyan]Available Episodes[/bold cyan] (showing {min(limit, len(episodes))} of {len(episodes)})\n")
            
            for idx, episode in enumerate(episodes[:limit], 1):
                self.console.print(f"{idx:3d}. {episode}")
            
            if len(episodes) > limit:
                self.console.print(f"\n[dim]...and {len(episodes) - limit} more[/dim]")
        else:
            self.console.print("[yellow]No episode information available[/yellow]")
    
    def _save_results(self, results, filepath):
        """Save results to JSON file"""
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.console.print(f"[green]✓[/green] Results saved to: {output_path}")
        except Exception as e:
            self.console.print(f"[red]✗[/red] Error saving results: {e}")
    
    def _save_topics(self, topics, entities, filepath):
        """Save topics and entities to JSON"""
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'topics': topics,
                'entities': {k: list(v) for k, v in entities.items()}
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.console.print(f"[green]✓[/green] Topics saved to: {output_path}")
        except Exception as e:
            self.console.print(f"[red]✗[/red] Error saving topics: {e}")
    
    def cmd_status(self):
        """Show system status"""
        self.console.print("\n[bold cyan]System Status[/bold cyan]\n")
        
        table = Table(show_header=False, box=None)
        table.add_column("Component", style="cyan", width=25)
        table.add_column("Status", style="white")
        
        # Data status
        if self.data_loaded:
            table.add_row("Data", "[green]✓ Loaded[/green]")
            table.add_row("  Records", str(len(self.processed_data)))
        else:
            table.add_row("Data", "[red]✗ Not loaded[/red]")
        
        # Embeddings status
        if self.embeddings_generated:
            table.add_row("Embeddings", "[green]✓ Generated[/green]")
            table.add_row("  Dimension", str(self.embeddings.shape[1]))
            table.add_row("  Count", str(len(self.embeddings)))
        else:
            table.add_row("Embeddings", "[red]✗ Not generated[/red]")
        
        # Cache status
        cache_file = Path('cache/embeddings.npy')
        if cache_file.exists():
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            table.add_row("Cache", f"[green]✓ Available ({size_mb:.1f} MB)[/green]")
        else:
            table.add_row("Cache", "[yellow]No cache[/yellow]")
        
        self.console.print(table)
        self.console.print()

class InteractiveMode(cmd.Cmd):
    """Interactive command prompt"""
    
    intro = """\n[bold cyan]🎙️  Podcast Insight Engine - Interactive Mode[/bold cyan]
Type 'help' for available commands or 'exit' to quit.\n"""
    prompt = '[bold cyan]podcast>[/bold cyan] '
    
    def __init__(self, cli):
        super().__init__()
        self.cli = cli
        console.print(self.intro)
    
    def do_search(self, arg):
        """Search for content: search <query> [--top N] [--min-sim F]"""
        if not arg:
            console.print("[red]Usage: search <query> [--top N] [--min-sim F][/red]")
            return
        
        args = arg.split()
        top_k = 5
        min_sim = 0.0
        query_parts = []
        
        i = 0
        while i < len(args):
            if args[i] == '--top' and i + 1 < len(args):
                top_k = int(args[i + 1])
                i += 2
            elif args[i] == '--min-sim' and i + 1 < len(args):
                min_sim = float(args[i + 1])
                i += 2
            else:
                query_parts.append(args[i])
                i += 1
        
        query = ' '.join(query_parts)
        self.cli.cmd_search(query, top_k=top_k, min_similarity=min_sim)
    
    def do_summarize(self, arg):
        """Summarize episode: summarize <episode_title> [--detailed]"""
        if not arg:
            console.print("[red]Usage: summarize <episode_title> [--detailed][/red]")
            return
        
        summary_type = "brief"
        if arg.endswith(" --detailed"):
            summary_type = "detailed"
            arg = arg.replace(" --detailed", "")
        
        self.cli.cmd_summarize(arg.strip(), summary_type=summary_type)
    
    def do_topics(self, arg):
        """Extract topics: topics [--num N] [--episode <name>]"""
        num = 10
        episode = None
        
        parts = arg.split()
        i = 0
        while i < len(parts):
            if parts[i] == '--num' and i + 1 < len(parts):
                num = int(parts[i + 1])
                i += 2
            elif parts[i] == '--episode' and i + 1 < len(parts):
                episode = parts[i + 1]
                i += 2
            else:
                i += 1
        
        self.cli.cmd_topics(num_topics=num, episode=episode)
    
    def do_stats(self, arg):
        """Show statistics: stats [--detailed] [--episode <name>]"""
        detailed = '--detailed' in arg
        episode = None
        
        if '--episode' in arg:
            parts = arg.split('--episode')
            if len(parts) > 1:
                episode = parts[1].strip().split()[0]
        
        self.cli.cmd_stats(episode=episode, detailed=detailed)
    
    def do_episodes(self, arg):
        """List episodes: episodes [N]"""
        limit = 20
        if arg.strip().isdigit():
            limit = int(arg.strip())
        self.cli.cmd_list_episodes(limit=limit)
    
    def do_status(self, arg):
        """Show system status: status"""
        self.cli.cmd_status()
    
    def do_exit(self, arg):
        """Exit interactive mode: exit"""
        console.print("[cyan]Goodbye![/cyan]")
        return True
    
    def do_quit(self, arg):
        """Exit interactive mode: quit"""
        return self.do_exit(arg)
    
    def do_help(self, arg):
        """Show help"""
        if not arg:
            console.print("\n[bold cyan]Available Commands:[/bold cyan]\n")
            commands = {
                'search': 'Search transcripts semantically',
                'summarize': 'Generate episode summary',
                'topics': 'Extract topics and entities',
                'stats': 'Show dataset statistics',
                'episodes': 'List available episodes',
                'status': 'Show system status',
                'help': 'Show this help message',
                'exit/quit': 'Exit interactive mode'
            }
            
            table = Table(show_header=False, box=None)
            table.add_column("Command", style="cyan", width=15)
            table.add_column("Description", style="white")
            
            for cmd, desc in commands.items():
                table.add_row(cmd, desc)
            
            console.print(table)
            console.print("\n[dim]Type 'help <command>' for detailed usage[/dim]\n")
        else:
            super().do_help(arg)
    
    def emptyline(self):
        """Do nothing on empty line"""
        pass

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='🎙️  Podcast Insight Engine - CLI Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load data and generate embeddings
  python cli.py init --dataset transcripts.csv
  python cli.py embed
  
  # Search
  python cli.py search "stories about forgiveness"
  python cli.py search "mental health" --top 10
  
  # Analysis
  python cli.py summarize --episode "Episode 742"
  python cli.py topics --num 20
  python cli.py stats --detailed
  
  # Interactive mode
  python cli.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize with dataset')
    init_parser.add_argument('--dataset', required=True, help='Path to CSV dataset')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings')
    embed_parser.add_argument('--data', help='Data file (uses cached if not specified)')
    embed_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search transcripts')
    search_parser.add_argument('query', nargs='?', help='Search query')
    search_parser.add_argument('--top', type=int, default=5, help='Number of results')
    search_parser.add_argument('--min-similarity', type=float, default=0.0, help='Min similarity')
    search_parser.add_argument('--output', help='Save results to file')
    
    # Summarize command
    summ_parser = subparsers.add_parser('summarize', help='Summarize episode')
    summ_parser.add_argument('--episode', required=True, help='Episode title')
    summ_parser.add_argument('--type', choices=['brief', 'detailed'], default='brief')
    summ_parser.add_argument('--use-api', action='store_true', help='Use Claude API')
    
    # Topics command
    topics_parser = subparsers.add_parser('topics', help='Extract topics')
    topics_parser.add_argument('--num', type=int, default=10, help='Number of topics')
    topics_parser.add_argument('--episode', help='Specific episode')
    topics_parser.add_argument('--output', help='Save to file')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    stats_parser.add_argument('--episode', help='Specific episode')
    stats_parser.add_argument('--detailed', action='store_true', help='Detailed stats')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Episodes command
    episodes_parser = subparsers.add_parser('episodes', help='List available episodes')
    episodes_parser.add_argument('--limit', type=int, default=20, help='Number to show')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = PodcastCLI()
    
    # Route to appropriate command
    if args.command == 'init':
        console.print("[cyan]Initializing system...[/cyan]")
        if cli.load_data(args.dataset):
            console.print("[green]✓[/green] Initialization complete")
            console.print("\n[bold]Next steps:[/bold]")
            console.print("  1. Generate embeddings: [cyan]python cli.py embed[/cyan]")
            console.print("  2. Search: [cyan]python cli.py search \"your query\"[/cyan]")
            console.print("  3. Or try interactive mode: [cyan]python cli.py interactive[/cyan]")
    
    elif args.command == 'embed':
        if args.data:
            cli.load_data(args.data)
        cli.generate_embeddings(batch_size=args.batch_size)
    
    elif args.command == 'search':
        if args.query:
            cli.cmd_search(args.query, top_k=args.top, min_similarity=args.min_similarity, output=args.output)
        else:
            console.print("[red]Error: Provide a search query[/red]")
            console.print("Example: [cyan]python cli.py search \"forgiveness stories\"[/cyan]")
    
    elif args.command == 'summarize':
        cli.cmd_summarize(args.episode, summary_type=args.type, use_api=args.use_api)
    
    elif args.command == 'topics':
        cli.cmd_topics(num_topics=args.num, episode=args.episode, output=args.output)
    
    elif args.command == 'stats':
        cli.cmd_stats(episode=args.episode, detailed=args.detailed)
    
    elif args.command == 'status':
        cli.cmd_status()
    
    elif args.command == 'episodes':
        cli.cmd_list_episodes(limit=args.limit)
    
    elif args.command == 'interactive':
        InteractiveMode(cli).cmdloop()

if __name__ == '__main__':
    main()