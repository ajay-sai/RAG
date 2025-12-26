# Quick Start Guide - RAG Strategy Lab

Welcome to the RAG Strategy Lab! This tool helps you learn and experiment with advanced Retrieval-Augmented Generation techniques.

## 🚀 Getting Started in 3 Steps

### Step 1: Setup (One-time)

```bash
# 1. Navigate to implementation folder
cd implementation

# 2. Install system dependencies
# Ubuntu/Debian:
sudo apt-get update && sudo apt-get install -y \
    ffmpeg \
    build-essential \
    gcc \
    postgresql-client \
    libpq-dev

# macOS:
brew install ffmpeg postgresql
xcode-select --install

# 3. Install Python dependencies
pip install -r requirements-advanced.txt
# or with uv:
uv sync

# Note: System dependencies are required for:
# - ffmpeg: Audio transcription with Whisper
# - build-essential/gcc: Compiling Python packages (psycopg2, etc.)
# - postgresql-client: Database command-line tools
# - libpq-dev: PostgreSQL development headers
```# 4. Setup PostgreSQL
sudo service postgresql start
sudo -u postgres psql -c "CREATE DATABASE ragdb;"
sudo -u postgres psql -c "CREATE USER raguser WITH PASSWORD 'ragpass123';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ragdb TO raguser;"
sudo -u postgres psql -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -d ragdb < sql/schema.sql
sudo -u postgres psql -d ragdb -c "GRANT ALL ON SCHEMA public TO raguser;"

# 4. Create .env file
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Step 2: Launch the App

```bash
cd implementation
streamlit run app.py
```

The app will open at http://localhost:8501

### Step 3: Try It Out!

1. **Ingestion Lab** (First Tab)
   - Upload 1-2 small documents (PDF, DOCX, or MD)
   - Or select from pre-loaded documents
   - Click "Run Ingestion Pipeline"
   - Wait for processing to complete

2. **Strategy Lab** (Second Tab)
   - Enter a question about your documents
   - Configure 1-3 different strategies
   - Click "Run Comparison"
   - Compare the results!

## 📚 What Can You Test?

### Chunking Strategies
- **Semantic:** Splits at natural text boundaries (best for most cases)
- **Fixed:** Equal-sized chunks (faster, good for uniform content)
- **Adaptive:** Structure-aware splitting (best for formatted documents)

### Retrieval Methods
- **Vector Search:** Traditional semantic similarity (baseline)
- **Multi-Query:** Expands your query into multiple variations
- **Hybrid:** Combines vector similarity + keyword search
- **Self-Reflective:** Iteratively improves search based on relevance

### Generation Styles
- **Standard:** Direct answer from retrieved content
- **Fact Verification:** Validates claims against sources
- **Multi-Hop Reasoning:** Breaks down complex questions
- **Uncertainty Estimation:** Provides confidence scores

### Other Options
- **Reranking:** Uses a Cross-Encoder to rerank results (better relevance, slower)
- **Embedding Models:** small (faster) vs large (better quality)
- **LLM Models:** gpt-4o-mini (cheaper) vs gpt-4o (more capable)

## 💡 Learning Tips

### Beginner Experiments
1. **Compare Baseline vs Multi-Query**
   - Strategy 1: Vector Search (baseline)
   - Strategy 2: Multi-Query
   - Same question in both
   - Notice how query expansion finds more relevant results

2. **Test Different Chunking**
   - Strategy 1: Semantic chunking
   - Strategy 2: Fixed chunking
   - Strategy 3: Adaptive chunking
   - See which works best for your documents

### Intermediate Experiments
1. **Hybrid vs Vector Search**
   - Compare pure semantic search with keyword-enhanced hybrid
   - Good for technical documents with specific terms

2. **Reranking Impact**
   - Strategy 1: Vector Search without reranking
   - Strategy 2: Vector Search with reranking
   - Measure the latency vs accuracy tradeoff

### Advanced Experiments
1. **Self-Reflective RAG**
   - Test with ambiguous or complex questions
   - Watch how it refines the search iteratively

2. **Multi-Hop Reasoning**
   - Ask questions that require combining information from multiple sources
   - Compare with standard generation

## 🎓 Educational Features

### Help Text
- Hover over any ℹ️ icon to see detailed explanations
- Each option includes trade-offs and recommendations

### Visual Indicators
- 🔍 Retrieval methods
- ✂️ Chunking strategies
- 🧠 Embedding models
- 🤖 LLM models
- 📝 Generation styles
- ⏱️ Execution time
- ⚡/⚖️/🐌 Cost estimates

### Tips Throughout
- 💡 "Tip for Learners" boxes with guidance
- Quick Start guide in sidebar
- Detailed help text on configuration options

## 🐛 Troubleshooting

### "DATABASE_URL is not set"
→ Create a `.env` file and add: `DATABASE_URL=postgresql://raguser:ragpass123@localhost:5432/ragdb`

### "OPENAI_API_KEY is not set"
→ Add your OpenAI API key to `.env`: `OPENAI_API_KEY=sk-...`
→ Get one from https://platform.openai.com/api-keys

### "extension 'vector' is not available"
→ Install pgvector: `sudo apt-get install postgresql-16-pgvector`

### "No module named 'transformers'"
→ Install dependencies: `pip install -r requirements-advanced.txt`

### App shows errors about missing modules
→ Make sure all dependencies are installed
→ Try: `pip install streamlit python-dotenv asyncpg pydantic-ai`

### PostgreSQL not running
→ Start it: `sudo service postgresql start`

### No documents found
→ Upload files using the "Upload Files" tab
→ Or add files manually to the `documents/` folder

## 📊 Performance Tips

### For Faster Testing
- Use gpt-4o-mini (faster and cheaper than gpt-4o)
- Use text-embedding-3-small (faster than large)
- Start with 1-2 small documents
- Disable contextual enrichment for initial tests

### For Better Quality
- Use gpt-4o for complex reasoning
- Use text-embedding-3-large for specialized domains
- Enable reranking for critical use cases
- Enable contextual enrichment for improved retrieval

## 🎨 UI Features

### Theme Toggle
- Click "🎨 Toggle Theme" to switch between light and dark modes
- Theme changes apply instantly

### File Management
- **Upload Files:** Drag & drop or browse to upload
- **Available Files:** Select which files to process
- File details show name and size

### Progress Tracking
- Progress bar shows ingestion status
- Status text updates during processing
- Detailed results after completion

### Strategy Comparison
- Side-by-side display of 3 strategies
- Color-coded results (green = success, red = error)
- Execution time and cost estimates
- Hover effects for better interactivity

## 📖 Further Reading

- **TESTING_GUIDE.md** - Comprehensive testing instructions
- **UI_CHANGES_SUMMARY.md** - Detailed list of all UI improvements
- **../docs/** - Detailed explanations of each RAG strategy
- **STRATEGIES.md** - Overview of all available strategies

## 🆘 Need Help?

1. Check the sidebar for warning messages
2. Review error messages - they include suggestions
3. Consult TESTING_GUIDE.md for detailed setup
4. Check the repository README for architecture details

## 🎯 Next Steps

Once you're comfortable with the basics:
1. Try all retrieval methods with the same query
2. Experiment with different chunking strategies
3. Test with your own domain-specific documents
4. Compare costs and latency across strategies
5. Learn when to use each technique in production

---

**Built for AI/ML and Data Science students and learners.**

Enjoy exploring advanced RAG strategies! 🚀
