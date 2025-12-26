# Screenshots

This folder contains screenshots demonstrating the RAG application features and capabilities.

## 📸 Screenshot Guidance

### For Contributors
- Add PNG or JPG screenshots showing key features
- Keep screenshots at ~1200px width for reasonable file sizes
- Use descriptive filenames (e.g., `cli-basic-search.png`, `ingestion-progress.png`)
- Reference in docs using: `![Description](path/to/screenshot.png)`

### Screenshot Categories

#### 1. Main Interface (`01-main-interface/`)
- CLI agent startup and welcome screen
- Interactive prompt examples
- Available strategies display

#### 2. Ingestion Process (`02-ingestion/`)
- Document ingestion progress
- Chunk creation statistics
- Database storage confirmation
- Example: Ingesting LoRA-SHIFT test paper

#### 3. Retrieval Strategies (`03-retrieval-strategies/`)
- Basic vector search results
- Re-ranking (two-stage retrieval)
- Multi-query RAG with query variations
- Self-reflective RAG with iterations
- Hybrid retrieval (dense + sparse)
- Fact verification with claim validation

#### 4. Testing (`04-testing/`)
- Test suite execution and results
- All 17 tests passing
- Coverage reports

#### 5. Performance Comparison (`05-performance/`)
- Strategy comparison tables
- Latency and cost metrics
- Relevance score comparisons

## 🎯 Key Features to Capture

### Query Examples with Different Strategies

**Sample Query:** "What is LoRA-SHIFT?"
- Screenshot with basic vector search (baseline)
- Screenshot with re-ranking (improved precision)
- Screenshot with multi-query (better coverage)

**Sample Query:** "How does LoRA-SHIFT improve over standard LoRA?"
- Show retrieved chunks with similarity scores
- Show generated response with source citations
- Show execution time and cost estimates

### Expected Outputs

When ingesting LoRA-SHIFT paper:
```
📄 Processing: LoRA-SHIFT-Final-Research-Paper.md
├─ Created 43 chunks
├─ Generated embeddings
└─ ✅ Stored in database
```

When running tests:
```
========================== 17 passed in 0.04s ==========================
```

## 📋 Alternative: Documentation Without Live Screenshots

Since screenshots require a live database and API keys, this repository provides:

1. **Detailed Feature Descriptions:** See this README for expected outputs
2. **Expected Output Examples:** Shown throughout this document
3. **Test Validation:** Automated tests prove functionality
4. **Code Examples:** Working code demonstrates all features

## 🚀 To Generate Screenshots

If you have access to live database and OpenAI API:

```bash
# 1. Setup
cd implementation
cp .env.example .env
# Add DATABASE_URL and OPENAI_API_KEY

# 2. Ingest test paper
python -m ingestion.ingest --documents documents/LoRA-SHIFT-Final-Research-Paper.md

# 3. Run agent
python rag_agent_advanced.py
# Try different queries and strategies

# 4. Run tests
pytest test_lora_shift_ingestion.py -v
# Screenshot the results

# 5. Save screenshots in appropriate folders
```

## 📖 Reference

For detailed expected outputs and feature descriptions without requiring live setup, see:
- **README.md** (this file) - Screenshot organization, guidance, and expected outputs
- **../../STUDENT_GUIDE.md** - Complete learning curriculum with practical examples
- **../../TROUBLESHOOTING.md** - Problem-solving guide

---

**Note:** The repository is fully functional and tested. Screenshots are optional visual aids. All functionality is validated through automated tests.
