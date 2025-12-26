# Getting Started: Quick Guide for New Users

Welcome! This guide helps you get started with the Advanced RAG Strategies repository.

---

## 🎯 What's New?

This repository has been comprehensively updated with:

✅ **Complete Test Suite** - 17 automated tests validating ingestion pipeline  
✅ **Student Learning Guide** - 9-week structured curriculum  
✅ **Troubleshooting Guide** - Solutions for common issues  
✅ **Test Research Paper** - LoRA-SHIFT paper for hands-on testing  
✅ **Enhanced Documentation** - Clear setup and usage instructions  

---

## 🚀 Quick Start (5 minutes)

### For Learning (No Setup Required)

**Start here if you want to learn RAG concepts first:**

1. **Read the Overview**
   ```bash
   # Open these files in order:
   README.md                  # Project overview
   STUDENT_GUIDE.md          # Learning curriculum
   docs/01-reranking.md      # Start with simplest strategy
   ```

2. **Explore Examples**
   ```bash
   cd examples
   cat 01_reranking.py       # < 50 line examples
   cat 05_query_expansion.py # See more strategies
   ```

3. **Run Tests (No Database Required)**
   ```bash
   cd implementation
   pip install pytest pytest-asyncio
   pytest test_lora_shift_ingestion.py -v
   # ✅ 17 tests should pass
   ```

4. **Study Test Paper**
   ```bash
   less implementation/documents/LoRA-SHIFT-Final-Research-Paper.md
   # 19,000+ character research paper
   ```

**Time:** 30-60 minutes  
**What you'll learn:** RAG concepts, strategies, testing patterns

---

### For Hands-On Testing (Requires Setup)

**Follow this if you want to run the actual application:**

#### Step 1: Prerequisites (10 minutes)

```bash
# 1. Python 3.9+
python --version  # Should be >= 3.9

# 2. PostgreSQL with pgvector
# Option A: Cloud (easiest)
# - Sign up for Neon (neon.tech) or Supabase (supabase.com)
# - pgvector comes pre-installed

# Option B: Local
sudo apt-get install postgresql-16 postgresql-16-pgvector
sudo service postgresql start
```

#### Step 2: Install Dependencies (5 minutes)

```bash
cd implementation

# Option A: Using pip
pip install -r requirements-advanced.txt

# Option B: Using uv (recommended)
pip install uv
uv sync
```

#### Step 3: Configure Environment (5 minutes)

```bash
# 1. Copy example
cp .env.example .env

# 2. Edit .env
nano .env  # or use your editor

# 3. Add these required variables:
DATABASE_URL=postgresql://user:pass@host:port/dbname
OPENAI_API_KEY=sk-...  # From platform.openai.com/api-keys
```

#### Step 4: Initialize Database (2 minutes)

```bash
# Run schema
psql $DATABASE_URL < sql/schema.sql

# Verify
psql $DATABASE_URL -c "\dt"
# Should show: documents, chunks
```

#### Step 5: Ingest Test Paper (2 minutes)

```bash
python -m ingestion.ingest --documents documents/LoRA-SHIFT-Final-Research-Paper.md

# Expected output:
# ✅ Processed 1 document
# ✅ Created ~43 chunks
# ✅ Stored in database
```

#### Step 6: Run the Agent! (1 minute)

```bash
python rag_agent_advanced.py

# Try these queries:
# - "What is LoRA-SHIFT?"
# - "How does LoRA-SHIFT improve over standard LoRA?"
# - "What datasets were used in experiments?"
```

**Total Time:** ~25 minutes  
**What you'll have:** Fully functional RAG system with test data

---

## 📚 Learning Path

### Beginner (Week 1-2)

**Goal:** Understand basic RAG pipeline

1. **Day 1-2:** Read STUDENT_GUIDE.md Week 1 section
2. **Day 3-4:** Study chunking (docs/07-context-aware-chunking.md)
3. **Day 5:** Run tests to understand ingestion
4. **Day 6-7:** Study re-ranking (docs/01-reranking.md)

**Outcome:** Understand: documents → chunks → embeddings → retrieval

### Intermediate (Week 3-5)

**Goal:** Master query enhancement

1. **Week 3:** Query expansion and multi-query RAG
2. **Week 4:** Hybrid retrieval (semantic + keyword)
3. **Week 5:** Re-ranking for precision

**Practice:** Compare strategies on same queries

### Advanced (Week 6-9)

**Goal:** Implement complex strategies

1. **Week 6-7:** Self-reflective RAG
2. **Week 8:** Multi-hop reasoning
3. **Week 9:** Fine-tuning and optimization

**Project:** Build domain-specific RAG system

---

## 🐛 Troubleshooting

### Quick Fixes

**"No module named X"**
```bash
pip install -r implementation/requirements-advanced.txt
```

**"DATABASE_URL not set"**
```bash
cp implementation/.env.example implementation/.env
# Edit .env and add DATABASE_URL
```

**"extension 'vector' is not available"**
```bash
# Cloud: Enable in SQL editor
psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Local: Install pgvector
sudo apt-get install postgresql-16-pgvector
```

**For detailed solutions:** See TROUBLESHOOTING.md

---

## 📖 Key Documents

| Document | Purpose | When to Read |
|----------|---------|--------------|
| `README.md` | Project overview | First |
| `STUDENT_GUIDE.md` | Learning curriculum | For structured learning |
| `TROUBLESHOOTING.md` | Problem solutions | When stuck |
| `GEMINI.md` | Architecture context | For implementation details |
| `docs/*.md` | Strategy explanations | To understand each technique |
| `examples/*.py` | Simple code examples | To see concepts in code |

---

## 🎯 Common Use Cases

### Use Case 1: Learning RAG
**Time:** 2-4 weeks  
**Path:** STUDENT_GUIDE.md → Examples → Implementation  
**Setup:** Optional (tests work without database)

### Use Case 2: Building a RAG System
**Time:** 1-2 days  
**Path:** Quick Start → Test with own documents  
**Setup:** Required (need database + API key)

### Use Case 3: Research & Comparison
**Time:** 1 week  
**Path:** Study docs → Compare strategies → Measure performance  
**Setup:** Required

### Use Case 4: Teaching RAG
**Time:** 9-week course  
**Path:** Use STUDENT_GUIDE as syllabus  
**Setup:** Optional for lectures, required for labs

---

## 🎓 What Makes This Repository Special

### 1. Three Levels of Content
- **Theory:** Detailed docs explaining "why"
- **Pseudocode:** Simple examples showing "how"
- **Implementation:** Working code demonstrating "real use"

### 2. Practical Test Data
- Real research paper (LoRA-SHIFT)
- 19,000+ characters of content
- Suitable for meaningful testing

### 3. Validated Quality
- 17 automated tests (all passing)
- Security scan clean
- Code review completed

### 4. Student-Focused
- 9-week learning path
- Practical exercises
- Project ideas
- Clear explanations

### 5. Production-Aware
- Best practices documented
- Performance considerations
- Security guidelines
- Troubleshooting guide

---

## 🚀 Next Steps

1. **Choose Your Path:**
   - Learning? → Start with STUDENT_GUIDE.md Week 1
   - Building? → Follow "Hands-On Testing" above
   - Teaching? → Review curriculum in STUDENT_GUIDE.md
   - Researching? → Study strategy docs in docs/

2. **Run Tests:**
   ```bash
   cd implementation
   pytest test_lora_shift_ingestion.py -v
   ```

3. **Join Community:**
   - Star the repository if helpful
   - Open issues for questions
   - Contribute improvements

4. **Share Feedback:**
   - What worked well?
   - What was confusing?
   - What's missing?

---

## 💡 Tips for Success

1. **Start Simple:** Don't try to understand everything at once
2. **Run Code:** Reading is good, running is better
3. **Experiment:** Modify examples and see what happens
4. **Ask Questions:** Use GitHub issues or discussions
5. **Take Notes:** Document your learning journey

---

## 📞 Getting Help

### Self-Service (Fast)
1. Check TROUBLESHOOTING.md
2. Review STUDENT_GUIDE.md
3. Search GitHub issues
4. Run tests to understand behavior

### Community Support
1. Open GitHub issue
2. Include error messages
3. Share what you've tried
4. Describe expected vs actual behavior

---

## ✅ Success Criteria

You'll know you're on track when you can:

- [ ] Explain what RAG is and why it's useful
- [ ] Run the test suite successfully
- [ ] Ingest the LoRA-SHIFT test paper
- [ ] Execute queries and get relevant results
- [ ] Compare different retrieval strategies
- [ ] Understand trade-offs (speed vs accuracy vs cost)
- [ ] Build a simple RAG system for your use case

---

**Welcome to the world of Advanced RAG! 🎉**

Start with the Quick Start above, and don't hesitate to explore. Every expert was once a beginner.

Happy learning! 🚀

---
