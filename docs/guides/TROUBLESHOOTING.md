# Troubleshooting Guide

**Common issues and solutions for the RAG implementation.**

---

## 🔧 Setup Issues

### Issue: "No module named 'pydantic_ai'"

**Symptoms:**
```
ImportError: No module named 'pydantic_ai'
```

**Solutions:**
```bash
# Option 1: Using pip
cd implementation
pip install -r requirements-advanced.txt

# Option 2: Using uv (recommended)
uv sync

# Option 3: Install specific package
pip install pydantic-ai>=0.7.4
```

---

### Issue: "extension 'vector' is not available"

**Symptoms:**
```
psycopg2.errors.UndefinedFile: extension "vector" is not available
```

**Solutions:**

**For Ubuntu/Debian:**
```bash
# Install pgvector extension
sudo apt-get update
sudo apt-get install postgresql-16-pgvector

# Restart PostgreSQL
sudo service postgresql restart

# Enable in your database
psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**For Mac:**
```bash
# Using Homebrew
brew install pgvector

# Restart PostgreSQL
brew services restart postgresql

# Enable in your database
psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**For Neon/Supabase (Cloud):**
- pgvector is pre-installed
- Enable via SQL editor:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

### Issue: "DATABASE_URL is not set"

**Symptoms:**
```
KeyError: 'DATABASE_URL'
# or
streamlit error: DATABASE_URL not configured
```

**Solution:**
```bash
# 1. Copy example environment file
cd implementation
cp .env.example .env

# 2. Edit .env file
nano .env  # or vim, code, etc.

# 3. Add your database URL:
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Examples:
# Local: postgresql://raguser:ragpass123@localhost:5432/ragdb
# Neon: postgresql://user:pass@ep-name.region.neon.tech/dbname
# Supabase: postgresql://postgres.[ref]:pass@aws-0-region.pooler.supabase.com:5432/postgres
```

---

### Issue: "OPENAI_API_KEY is not set"

**Symptoms:**
```
OpenAI API error: Incorrect API key provided
# or
Warning: OPENAI_API_KEY not configured
```

**Solution:**
```bash
# 1. Get API key from https://platform.openai.com/api-keys

# 2. Add to .env file
echo "OPENAI_API_KEY=sk-..." >> .env

# 3. Verify
cat .env | grep OPENAI_API_KEY

# 4. Restart application
```

**Important**: Keep your API key secret! Never commit .env to git.

---

### Issue: "relation 'documents' does not exist"

**Symptoms:**
```
psycopg2.errors.UndefinedTable: relation "documents" does not exist
```

**Solution:**
```bash
# Run schema file to create tables
cd implementation
psql $DATABASE_URL < sql/schema.sql

# Or manually:
psql $DATABASE_URL
> \i sql/schema.sql
> \q
```

**Verify tables exist:**
```bash
psql $DATABASE_URL -c "\dt"
# Should show: documents, chunks
```

---

## 📊 Ingestion Issues

### Issue: Documents not being ingested

**Symptoms:**
- No error, but no documents in database
- "No documents found" message

**Debug steps:**
```bash
# 1. Check documents folder
ls -la implementation/documents/
# Should show your files

# 2. Check file permissions
ls -l implementation/documents/*.pdf
# Should be readable

# 3. Run ingestion with verbose output
cd implementation
python -m ingestion.ingest --documents ./documents --verbose

# 4. Check database
psql $DATABASE_URL -c "SELECT COUNT(*) FROM documents;"
psql $DATABASE_URL -c "SELECT COUNT(*) FROM chunks;"
```

**Common causes:**
- Files have wrong permissions (chmod 644)
- Unsupported file format
- Missing ffmpeg for audio files

---

### Issue: Missing system dependencies

**Symptoms:**
```
# For ffmpeg:
ERROR - Audio tranciption has an error: [Errno 2] No such file or directory: 'ffmpeg'

# For gcc/build tools:
error: command 'gcc' failed: No such file or directory
ERROR: Failed building wheel for psycopg2

# For PostgreSQL client:
psql: command not found

# For libpq:
Error: pg_config executable not found
```

**Solution - Install ALL system dependencies:**

**For Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y \
    ffmpeg \
    build-essential \
    gcc \
    postgresql-client \
    libpq-dev

# Verify installations
ffmpeg -version
gcc --version
psql --version
pg_config --version
```

**For macOS:**
```bash
# Install packages
brew install ffmpeg postgresql

# Install Xcode Command Line Tools
xcode-select --install

# Verify installations
ffmpeg -version
gcc --version
psql --version
```

**For Windows:**
1. **ffmpeg**: Download from https://ffmpeg.org/download.html and add to PATH
2. **PostgreSQL**: Download from https://www.postgresql.org/download/windows/
3. **Build Tools**: Install Visual Studio Build Tools from https://visualstudio.microsoft.com/downloads/

**Docker users:** All dependencies are included in the Dockerfile

---

### Issue: "[Errno 2] No such file or directory: 'ffmpeg'"

**Symptoms:**
```
ERROR - Audio tranciption has an error: [Errno 2] No such file or directory: 'ffmpeg'
WARNING - ❌ Failed to transcribe with WHISPER_TINY
```

**Cause:** Audio files (MP3, WAV) require ffmpeg for transcription, but it's not installed.

**Solution:**

**For Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg

# Verify installation
ffmpeg -version
```

**For macOS:**
```bash
brew install ffmpeg

# Verify installation
ffmpeg -version
```

**For Windows:**
1. Download from https://ffmpeg.org/download.html
2. Extract and add to PATH
3. Restart terminal
4. Verify: `ffmpeg -version`

**Docker users:** ffmpeg is already included in the Dockerfile (line 6)

**After installation:**
```bash
# Re-run ingestion
cd implementation
python -m ingestion.ingest --documents ./documents
```

---

### Issue: Audio files not transcribed

**Symptoms:**
- Audio files processed but only create placeholder chunks
- No actual transcription content

**Debug steps:**
```bash
# 1. Verify ffmpeg is installed
ffmpeg -version

# 2. Check if Whisper model loads
python -c "import whisper; print(whisper.available_models())"

# 3. Check system resources
python -c "from ingestion.resource_monitor import ResourceMonitor; ResourceMonitor.print_resource_summary()"

# 4. Try with smaller Whisper model
python -m ingestion.ingest --documents ./documents --mode light
```

**Common causes:**
- ffmpeg not installed (see above)
- Insufficient RAM for Whisper model (use --mode light)
- Corrupted audio files
- Unsupported audio codec

---
1. Empty documents folder
2. Unsupported file formats (use PDF, DOCX, MD, TXT)
3. Corrupted files
4. Insufficient permissions

---

### Issue: "Docling failed to process document"

**Symptoms:**
```
Error processing document X: Docling conversion failed
```

**Solutions:**

1. **Check file format:**
```bash
file documents/myfile.pdf
# Should show: PDF document, version X.X
```

2. **Try without Docling:**
```bash
# Use simple chunker for text files
python -m ingestion.ingest --documents ./documents --chunker fixed
```

3. **Check file isn't corrupted:**
```bash
# Try opening in another application
# For PDFs:
pdfinfo documents/myfile.pdf
```

4. **Reinstall Docling:**
```bash
pip uninstall docling
pip install "docling[vlm]>=2.55.0"
```

---

### Issue: Out of memory during ingestion

**Symptoms:**
```
MemoryError
# or
Killed (process terminated)
```

**Solutions:**

1. **Process fewer documents at once:**
```bash
# Instead of all documents:
python -m ingestion.ingest --documents ./documents

# Process one at a time:
python -m ingestion.ingest --documents ./documents/file1.pdf
python -m ingestion.ingest --documents ./documents/file2.pdf
```

2. **Use smaller chunks:**
```python
# In .env or config
CHUNK_SIZE=256  # Instead of default 512
```

3. **Increase system memory or use smaller files**

---

### Issue: Embedding generation is slow

**Symptoms:**
- Ingestion takes very long (> 5 minutes per document)

**Solutions:**

1. **Use local embeddings (faster, no API calls):**
```python
# In embedder.py, use sentence-transformers instead of OpenAI
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

2. **Batch embeddings:**
```python
# Already implemented in embedder.py
# Processes multiple chunks at once
```

3. **Check API rate limits:**
```bash
# OpenAI has rate limits (3000 RPM for free tier)
# Space out requests or upgrade plan
```

---

## 🔍 Retrieval Issues

### Issue: No results returned for queries

**Symptoms:**
```
No relevant documents found
```

**Debug steps:**

1. **Verify data exists:**
```sql
-- Check chunks exist with embeddings
SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL;

-- Should return > 0
```

2. **Test vector search function:**
```sql
-- Create a test embedding
SELECT match_chunks(
  ARRAY[0.1, 0.2, ...]::vector(1536),  -- Test vector
  5
);
```

3. **Check embedding dimensions:**
```sql
-- Should be 1536 for OpenAI text-embedding-3-small
SELECT 
  id, 
  vector_dims(embedding) as dims 
FROM chunks 
LIMIT 5;
```

**Common causes:**
- Chunks ingested without embeddings
- Wrong embedding dimensions
- Vector index not created

**Fix:**
```bash
# Re-run ingestion
python -m ingestion.ingest --documents ./documents
```

---

### Issue: Results not relevant

**Symptoms:**
- Retrieval returns chunks that don't match query
- Low similarity scores (< 0.5)

**Solutions:**

1. **Try different retrieval strategies:**
```python
# Instead of basic vector search:
await agent.run("What is X?", tool_choice="search_with_multi_query")
# or
await agent.run("What is X?", tool_choice="search_with_reranking")
```

2. **Improve chunking:**
```bash
# Use semantic chunking
python -m ingestion.ingest --documents ./documents --chunker semantic

# Or adaptive chunking
python -m ingestion.ingest --documents ./documents --chunker adaptive
```

3. **Enable contextual enrichment:**
```bash
# Adds document context to chunks
python -m ingestion.ingest --documents ./documents --contextual
```

4. **Check if query needs expansion:**
```python
# Vague query: "Tell me about X"
# Better: "What is X, how does it work, and what are its advantages?"
```

---

### Issue: Slow retrieval (> 2 seconds)

**Symptoms:**
- Queries take > 2 seconds to complete
- High database latency

**Solutions:**

1. **Check vector index:**
```sql
-- Should have index on embedding column
\d chunks
-- Look for: idx_chunks_embedding

-- If missing, create:
CREATE INDEX idx_chunks_embedding 
ON chunks USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

2. **Use smaller result limits:**
```python
# Instead of returning 20 results:
results = await search_knowledge_base(query, limit=5)
```

3. **Optimize database:**
```sql
-- Analyze tables for query planning
ANALYZE chunks;
ANALYZE documents;

-- Vacuum to reclaim space
VACUUM ANALYZE chunks;
```

4. **Use connection pooling:**
```python
# Already implemented in db_utils.py
# Reuses database connections
```

---

## 🤖 Agent Issues

### Issue: Agent doesn't call tools

**Symptoms:**
- Agent responds without searching knowledge base
- No tool calls logged

**Debug:**
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check tool definitions
print(agent.tools)
```

**Solutions:**

1. **Make tool purposes clear:**
```python
@agent.tool
async def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for information.
    USE THIS TOOL whenever user asks about: documents, data, facts.
    """
```

2. **Use explicit tool choice:**
```python
# Force tool usage
result = await agent.run(
    user_message,
    tool_choice="search_knowledge_base"
)
```

3. **Check system prompt:**
```python
# System prompt should mention tools
system_prompt = """
You are a helpful assistant with access to a knowledge base.
Always search the knowledge base before answering questions.
"""
```

---

### Issue: Agent hallucinates information

**Symptoms:**
- Responses include information not in documents
- Made-up facts or "creative" answers

**Solutions:**

1. **Stricter system prompt:**
```python
system_prompt = """
You are a factual assistant. CRITICAL RULES:
1. ONLY use information from retrieved documents
2. If information is not in documents, say "I don't have that information"
3. NEVER make assumptions or fill in gaps
4. Always cite sources: [Source: document_name]
"""
```

2. **Enable fact verification:**
```python
# Use fact verification strategy
result = await agent.run(
    query,
    tool_choice="search_with_fact_verification"
)
```

3. **Lower temperature:**
```python
# More deterministic, less creative
agent = Agent(
    'openai:gpt-4o-mini',
    temperature=0.1  # Instead of default 0.7
)
```

4. **Add source citation requirement:**
```python
@agent.tool
async def search_knowledge_base(query: str) -> str:
    """Returns: Formatted results with [Source: title] citations"""
    # Implementation ensures every chunk includes source info
```

---

## 🐍 Python Environment Issues

### Issue: ModuleNotFoundError

**Symptoms:**
```
ModuleNotFoundError: No module named 'X'
```

**Solution:**
```bash
# Option 1: Reinstall all dependencies
cd implementation
pip install -r requirements-advanced.txt --force-reinstall

# Option 2: Check Python version
python --version
# Should be 3.9 or later

# Option 3: Check virtual environment
which python
# Should point to venv/bin/python if using venv

# Option 4: Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements-advanced.txt
```

---

### Issue: "asyncio" related errors

**Symptoms:**
```
RuntimeError: Event loop is closed
# or
RuntimeError: This event loop is already running
```

**Solutions:**

1. **Use asyncio.run():**
```python
# Correct:
asyncio.run(main())

# Incorrect:
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

2. **Don't mix sync and async:**
```python
# Incorrect:
def my_function():
    result = await some_async_function()  # Can't await in sync function

# Correct:
async def my_function():
    result = await some_async_function()
```

---

## 🌐 Network Issues

### Issue: "Connection refused" to database

**Symptoms:**
```
psycopg2.OperationalError: could not connect to server: Connection refused
```

**Solutions:**

1. **Check PostgreSQL is running:**
```bash
# Linux
sudo service postgresql status
sudo service postgresql start

# Mac
brew services list
brew services start postgresql

# Docker
docker ps | grep postgres
docker start postgres-container
```

2. **Check connection string:**
```bash
# Format: postgresql://user:password@host:port/database
# Common mistakes:
# - Wrong port (5432 is default)
# - Wrong host (localhost vs 127.0.0.1)
# - Special characters in password need URL encoding
```

3. **Test connection:**
```bash
psql $DATABASE_URL -c "SELECT 1"
# Should return: 1
```

---

### Issue: OpenAI API errors

**Symptoms:**
```
openai.error.RateLimitError: Rate limit exceeded
# or
openai.error.APIError: Service unavailable
```

**Solutions:**

1. **Rate limiting:**
```python
# Add retry logic with backoff
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def call_openai_api(...):
    # Your API call
```

2. **Check usage limits:**
```bash
# Visit https://platform.openai.com/usage
# Check your tier limits and current usage
```

3. **Use fallback:**
```python
try:
    response = await openai_call()
except openai.error.RateLimitError:
    # Fallback to cached results or simpler strategy
    response = await cached_search(query)
```

---

## 📱 Streamlit UI Issues

### Issue: UI not loading

**Symptoms:**
```
streamlit run app.py
# But browser shows nothing or error
```

**Solutions:**

1. **Check port:**
```bash
# Default is 8501
streamlit run app.py --server.port 8501

# Try different port
streamlit run app.py --server.port 8502
```

2. **Check firewall:**
```bash
# Allow port 8501
sudo ufw allow 8501
```

3. **Access via correct URL:**
```
# Typical URL (default on most systems)
http://localhost:8501
# If localhost does not work, try the loopback IP instead (behavior can vary by system)
http://127.0.0.1:8501
```

---

### Issue: "Session state" errors

**Symptoms:**
```
StreamlitAPIException: Session state X doesn't exist
```

**Solution:**
```python
# Always check before accessing
if 'my_variable' not in st.session_state:
    st.session_state.my_variable = default_value

# Or use get() with default
value = st.session_state.get('my_variable', default_value)
```

---

## 🧪 Testing Issues

### Issue: Tests failing

**Symptoms:**
```
pytest test_*.py
# Some tests fail
```

**Debug:**
```bash
# Run with verbose output
pytest test_lora_shift_ingestion.py -v -s

# Run single test
pytest test_lora_shift_ingestion.py::TestLoRAShiftIngestion::test_paper_exists -v

# Show print statements
pytest test_lora_shift_ingestion.py -v -s --log-cli-level=DEBUG
```

**Common issues:**
1. Missing test dependencies: `pip install pytest pytest-asyncio`
2. Database not available: Tests should mock database calls
3. Environment variables: Tests shouldn't require .env

---

## 🔐 Security Issues

### Issue: API key exposed

**Symptoms:**
- Accidentally committed .env to git
- API key visible in logs

**Immediate actions:**

1. **Revoke compromised key:**
```
Visit https://platform.openai.com/api-keys
Delete the exposed key
Generate a new one
```

2. **Remove from git history:**
```bash
# If just committed
git reset --soft HEAD~1
git restore --staged .env

# If pushed
# Consider rewriting history (dangerous!)
# Or just revoke key and move on
```

3. **Add to .gitignore:**
```bash
echo ".env" >> .gitignore
git add .gitignore
git commit -m "Add .env to gitignore"
```

4. **Use environment variables:**
```bash
# Instead of .env file in production
export OPENAI_API_KEY=sk-...
export DATABASE_URL=postgresql://...
```

---

## 📝 Getting More Help

### Enable Debug Logging

```python
# Add to your script
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Check System Status

```bash
# Python version
python --version

# Installed packages
pip list | grep -E "(pydantic|openai|asyncpg|docling)"

# Database status
psql $DATABASE_URL -c "SELECT version();"

# Disk space
df -h

# Memory
free -h
```

### Report Issues

When reporting bugs, include:

1. **Error message** (full traceback)
2. **Steps to reproduce**
3. **Environment info**:
   ```bash
   python --version
   pip list
   uname -a  # or OS version
   ```
4. **Configuration** (without secrets!)
5. **Expected vs actual behavior**

---

## 📚 Additional Resources

- **PostgreSQL Docs**: https://www.postgresql.org/docs/
- **pgvector Guide**: https://github.com/pgvector/pgvector
- **Pydantic AI**: https://ai.pydantic.dev/
- **OpenAI API**: https://platform.openai.com/docs
- **Docling**: https://github.com/DS4SD/docling

---

**Still stuck? Check:**
1. GitHub Issues for this repo
2. Stack Overflow with error message
3. Discord/Slack communities for RAG/LLMs

---

**End of Troubleshooting Guide**
