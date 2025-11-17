# Quick Start Guide

Get the Podcast Insight Engine running in 5 minutes!

## Step 1: Install Dependencies

### Windows
```bash
install.bat
```

### Linux/Mac
```bash
chmod +x install.sh
./install.sh
```

## Step 2: Test the System

Run the test script to verify everything works:

```bash
python test_system.py
```

You should see:
```
ALL TESTS PASSED! ✓
```

## Step 3: Run the Application

### Option A: Web Interface (Recommended)

```bash
streamlit run main.py
```

Your browser will open to `http://localhost:8501`

### Option B: Command Line

```bash
# Search example
python run.py --data your_data.csv --mode search --query "forgiveness"

# Summarize example
python run.py --data your_data.csv --mode summarize --episode "Episode Title"
```

## Step 4: Upload Your Data

In the web interface:
1. Click "Browse files" in the sidebar
2. Select your CSV file
3. Click "Generate Embeddings"
4. Start searching!

## Expected CSV Format

Your CSV needs at minimum a text column:

```csv
text
"This is the transcript of the podcast episode..."
"More transcript content here..."
```

Better with episode info:

```csv
episode_title,speaker,text
"Episode 1",Host,"Welcome to the show..."
"Episode 1",Guest,"Thanks for having me..."
```

## Example Queries

Try these searches:
- "stories about forgiveness"
- "episodes featuring immigrants"
- "conversations about mental health"
- "childhood memories"

## Troubleshooting

**Problem:** `ModuleNotFoundError`
**Solution:** Run `pip install -r requirements.txt`

**Problem:** spaCy model not found
**Solution:** Run `python -m spacy download en_core_web_sm`

**Problem:** Out of memory
**Solution:** Reduce batch size in `config.py` or process fewer segments

**Problem:** Slow performance
**Solution:** Set `USE_GPU = True` in `config.py` if you have a GPU

## Next Steps

- Check `README.md` for detailed documentation
- Read `config.py` to customize settings
- Review your project plan document for understanding the system architecture
- Add your feedback mechanisms to improve results

## Support

Having issues? Check:
1. All dependencies installed correctly
2. Python version is 3.8 or higher
3. CSV file format matches expectations
4. Sufficient disk space for embeddings cache

For detailed help, see `README.md` or the project plan document.
