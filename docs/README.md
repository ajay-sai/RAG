# Documentation Index

Welcome to the RAG Strategies documentation! This folder contains all project documentation organized into clear categories.

## 📂 Folder Structure

```
docs/
├── README.md (this file)
├── guides/           # User guides and learning resources
├── implementation/   # Technical implementation documentation  
├── project/          # Project management and planning docs
└── strategies/       # Individual strategy documentation (01-16)
```

---

## 📚 Quick Navigation

### 🎓 For Learners

Start here if you're learning about RAG strategies:

- **[guides/GETTING_STARTED.md](guides/GETTING_STARTED.md)** - Quick start guide (5 min setup)
- **[guides/STUDENT_GUIDE.md](guides/STUDENT_GUIDE.md)** - 9-week structured learning curriculum
- **[guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)** - Solutions for common issues

### 💻 For Developers

Implementation guides and technical details:

- **[implementation/QUICK_START.md](implementation/QUICK_START.md)** - Get the app running in 3 steps
- **[implementation/STRATEGIES.md](implementation/STRATEGIES.md)** - Overview of all implemented strategies
- **[implementation/IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md)** - Detailed implementation reference
- **[implementation/TESTING_GUIDE.md](implementation/TESTING_GUIDE.md)** - How to test the application
- **[implementation/README_UI.md](implementation/README_UI.md)** - UI features and usage

### 🧪 For Experimenters

Use the Strategy Lab to compare techniques:

- **[implementation/UI_CHANGES_SUMMARY.md](implementation/UI_CHANGES_SUMMARY.md)** - Recent UI improvements
- **[implementation/FIXES_README.md](implementation/FIXES_README.md)** - Complete list of fixes
- **[implementation/IMPLEMENTATION_SUMMARY.md](implementation/IMPLEMENTATION_SUMMARY.md)** - What's been built

### 🏗️ For Project Contributors

Project planning and coordination:

- **[project/PROJECT_NOTES.md](project/PROJECT_NOTES.md)** - Design decisions and task tracker
- **[project/GEMINI.md](project/GEMINI.md)** - Context for AI assistants
- **[project/FINAL_QA_SUMMARY.md](project/FINAL_QA_SUMMARY.md)** - Quality check summary

---

## 📖 Strategy Documentation

Detailed theory, research, and implementation for each RAG strategy:

| # | Strategy | Doc Link | Status |
|---|----------|----------|--------|
| 01 | Re-ranking | [01-reranking.md](01-reranking.md) | ✅ Complete |
| 02 | Agentic RAG | [02-agentic-rag.md](02-agentic-rag.md) | ✅ Complete |
| 03 | Knowledge Graphs | [03-knowledge-graphs.md](03-knowledge-graphs.md) | ✅ Complete |
| 04 | Contextual Retrieval | [04-contextual-retrieval.md](04-contextual-retrieval.md) | ✅ Complete |
| 05 | Query Expansion | [05-query-expansion.md](05-query-expansion.md) | ✅ Complete |
| 06 | Multi-Query RAG | [06-multi-query-rag.md](06-multi-query-rag.md) | ✅ Complete |
| 07 | Context-Aware Chunking | [07-context-aware-chunking.md](07-context-aware-chunking.md) | ✅ Complete |
| 08 | Late Chunking | [08-late-chunking.md](08-late-chunking.md) | ✅ Complete |
| 09 | Hierarchical RAG | [09-hierarchical-rag.md](09-hierarchical-rag.md) | ✅ Complete |
| 10 | Self-Reflective RAG | [10-self-reflective-rag.md](10-self-reflective-rag.md) | ✅ Complete |
| 11 | Fine-Tuned Embeddings | [11-fine-tuned-embeddings.md](11-fine-tuned-embeddings.md) | ✅ Complete |
| 12 | Hybrid Retrieval | [12-hybrid-retrieval.md](12-hybrid-retrieval.md) | ✅ Complete |
| 13 | Fact Verification | [13-fact-verification.md](13-fact-verification.md) | ✅ Complete |
| 14 | Multi-Hop Reasoning | [14-multi-hop-reasoning.md](14-multi-hop-reasoning.md) | ✅ Complete |
| 15 | Uncertainty Calibration | [15-uncertainty-calibration.md](15-uncertainty-calibration.md) | ✅ Complete |
| 16 | Adaptive Chunking | [16-adaptive-chunking.md](16-adaptive-chunking.md) | ✅ Complete |

---

## 🎯 Recommended Learning Paths

### Path 1: Quick Exploration (1 hour)
1. [guides/GETTING_STARTED.md](guides/GETTING_STARTED.md)
2. [01-reranking.md](01-reranking.md) 
3. [05-query-expansion.md](05-query-expansion.md)
4. Run tests: `cd implementation && python test_lora_shift_ingestion.py`

### Path 2: Deep Dive (1 week)
1. [guides/STUDENT_GUIDE.md](guides/STUDENT_GUIDE.md) - Week 1
2. Set up environment: [guides/GETTING_STARTED.md](guides/GETTING_STARTED.md)
3. Study strategies 01-07 (fundamentals)
4. Run the app: [implementation/QUICK_START.md](implementation/QUICK_START.md)

### Path 3: Implementation Focus (3 days)
1. [implementation/QUICK_START.md](implementation/QUICK_START.md)
2. [implementation/STRATEGIES.md](implementation/STRATEGIES.md)
3. [implementation/IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md)
4. Experiment in Strategy Lab

---

## 🔍 Finding Specific Information

**Setup Issues?** → [guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)

**Learning RAG?** → [guides/STUDENT_GUIDE.md](guides/STUDENT_GUIDE.md)

**Understanding a strategy?** → Strategy docs (01-16)

**Implementing a feature?** → [implementation/IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md)

**Testing the app?** → [implementation/TESTING_GUIDE.md](implementation/TESTING_GUIDE.md)

**UI questions?** → [implementation/README_UI.md](implementation/README_UI.md)

**Contributing?** → [project/PROJECT_NOTES.md](project/PROJECT_NOTES.md)

---

## 📝 Document Categories Explained

### `/guides/` - User-Facing Documentation
Educational content for students, practitioners, and learners. No code expertise required.

### `/implementation/` - Technical Documentation  
Developer-focused docs with code examples, architecture decisions, and implementation details.

### `/project/` - Project Management
Planning documents, design decisions, task tracking, and AI assistant context.

### Root Strategy Docs (01-16)
Deep-dive theory and research papers for each RAG strategy with citations and comparisons.

---

## 🤝 Contributing

When adding new documentation:
- User guides → `guides/`
- Implementation details → `implementation/`
- Strategy explanations → Root `docs/` (numbered)
- Project planning → `project/`

---

**Need help?** Check [guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md) or open an issue!
