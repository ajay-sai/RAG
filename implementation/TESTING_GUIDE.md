# Testing Guide for Streamlit RAG App Updates

## Changes Made

### 1. ✅ Theme Toggle Bug Fix
- **Issue**: Theme toggle was not updating the UI instantly
- **Fix**: Added `st.rerun()` to `toggle_theme()` function to force immediate UI refresh
- **Location**: `app.py` line 46
- **Test**: Click the "🎨 Toggle Theme" button and verify the theme changes instantly

### 2. ✅ Lab Order Fix
- **Issue**: Strategy Lab was shown first instead of Ingestion Lab
- **Fix**: Reordered navigation radio button options and updated index
- **Location**: `app.py` line 490
- **Test**: Open app and verify "Ingestion Lab" is selected by default

### 3. ✅ File Upload Feature
- **Issue**: No way for users to upload their own files
- **Fix**: Added file upload component with tabs for "Upload Files" and "Available Files"
- **Features**:
  - Support for PDF, DOCX, MD, TXT, MP3, WAV files
  - Save uploaded files to documents folder
  - View and select from available files
  - Select/deselect files for ingestion
  - File details (name, size)
- **Location**: `app.py` lines 270-336
- **Test**: 
  1. Go to Ingestion Lab
  2. Click "Upload Files" tab
  3. Upload test files
  4. Click "💾 Save Uploaded Files"
  5. Switch to "Available Files" tab and verify files are shown
  6. Select/deselect files using checkboxes

### 4. ✅ Strategy Lab - Chunking Selection Fix
- **Issue**: Chunking strategy was not selectable (disabled text input)
- **Fix**: Changed from disabled text_input to selectable selectbox
- **Location**: `app.py` lines 508-517
- **Test**: 
  1. Go to Strategy Lab
  2. Verify "Chunking" and "Embedding" are now dropdowns, not disabled text fields
  3. Change chunking strategy in each strategy column
  4. Verify different strategies can have different chunking methods

### 5. ✅ UI/UX Improvements
- **Enhanced CSS Styling**:
  - Better shadows and borders for containers
  - Hover effects on strategy cards
  - Improved spacing and padding
  - Better metric tag styling with rounded corners
  - Improved button hover effects
  - **Location**: `app.py` lines 81-158

- **Better Help Text**:
  - Comprehensive tooltips for all configuration options
  - Emoji indicators for visual clarity
  - Explanations of trade-offs for learners
  - **Location**: `app.py` throughout

- **Improved Instructions**:
  - Welcome message and quick start guide in sidebar
  - Educational descriptions for each lab
  - Tips for learners
  - Better context for configuration options
  - **Location**: `app.py` lines 475-491, 264-272, 493-502

- **Enhanced Layout**:
  - Better organized file selection
  - Tabbed interface for upload vs. selection
  - Expandable file details
  - Progress bars and status indicators
  - Better result display with success/error indicators

### 6. ✅ Error Handling & Environment Checks
- **Features**:
  - Check for DATABASE_URL and OPENAI_API_KEY
  - Display warnings/errors in sidebar
  - Graceful degradation when dependencies missing
  - Helpful error messages with installation instructions
- **Location**: `app.py` lines 12-45, 475-491
- **Test**: 
  1. Start app without .env file and verify error message
  2. Add .env with DATABASE_URL but no OPENAI_API_KEY and verify warning

### 7. ✅ Ingestion Pipeline Updates
- **Issue**: No way to select specific files for ingestion
- **Fix**: Added `specific_files` parameter to `ingest_documents()` method
- **Location**: `ingestion/ingest.py` lines 95-125
- **Test**: 
  1. Select only 2-3 files from available files
  2. Run ingestion
  3. Verify only selected files are processed

## Testing Prerequisites

### Required Services
1. **PostgreSQL with pgvector**
   ```bash
   # Start PostgreSQL
   sudo service postgresql start
   
   # Create database and user (if not exists)
   sudo -u postgres psql -c "CREATE DATABASE ragdb;"
   sudo -u postgres psql -c "CREATE USER raguser WITH PASSWORD 'ragpass123';"
   sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ragdb TO raguser;"
   sudo -u postgres psql -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
   
   # Apply schema
   cd implementation
   sudo -u postgres psql -d ragdb < sql/schema.sql
   sudo -u postgres psql -d ragdb -c "GRANT ALL ON SCHEMA public TO raguser; GRANT ALL ON ALL TABLES IN SCHEMA public TO raguser; GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO raguser;"
   ```

2. **Environment Variables**
   ```bash
   # Create .env file
   cd implementation
   cp .env.example .env
   
   # Edit .env and add:
   # - DATABASE_URL=postgresql://raguser:ragpass123@localhost:5432/ragdb
   # - OPENAI_API_KEY=your-actual-openai-api-key
   ```

3. **Python Dependencies**
   ```bash
   cd implementation
   pip install -r requirements-advanced.txt
   # or
   uv sync
   ```

## Running the App

```bash
cd implementation
streamlit run app.py
```

The app should open in your browser at http://localhost:8501

## Manual Testing Checklist

### Theme Toggle
- [ ] Click "🎨 Toggle Theme" button
- [ ] Verify theme changes instantly from light to dark or vice versa
- [ ] Verify all components update colors correctly
- [ ] Click again to verify toggling back works

### Navigation & Lab Order
- [ ] Verify "Ingestion Lab" is the first tab and selected by default
- [ ] Click "Strategy Lab" and verify it switches
- [ ] Click back to "Ingestion Lab" and verify it switches

### Ingestion Lab - File Upload
- [ ] Go to "Upload Files" tab
- [ ] Click "Browse files" and upload 1-2 test documents (PDF, DOCX, or MD)
- [ ] Verify files appear in the list with correct names and sizes
- [ ] Click "💾 Save Uploaded Files"
- [ ] Verify success message appears
- [ ] Go to "Available Files" tab
- [ ] Verify uploaded files now appear in the list

### Ingestion Lab - File Selection
- [ ] In "Available Files" tab, verify all files are selected by default
- [ ] Uncheck "Select All Files"
- [ ] Manually select 2-3 files using the multiselect
- [ ] Expand "View File Details"
- [ ] Verify selected files show ✅ and unselected show ⬜

### Ingestion Lab - Configuration
- [ ] Verify all sliders work (Chunk Size, Chunk Overlap)
- [ ] Verify all dropdowns work (Chunker Type, Embedding Model)
- [ ] Hover over help icons (ℹ️) and verify tooltips appear
- [ ] Check "Use Contextual Enrichment" and verify it toggles

### Ingestion Lab - Pipeline Execution
**Note**: Requires valid OPENAI_API_KEY

- [ ] Select 1-2 small files
- [ ] Click "🔄 Run Ingestion Pipeline"
- [ ] Verify progress bar appears and updates
- [ ] Verify status text shows "Processing document X/Y..."
- [ ] Verify success message appears with number of documents processed
- [ ] Verify results show file names and chunk counts

### Strategy Lab - Strategy Configuration
- [ ] Verify you can see 3 strategy columns
- [ ] In each strategy column, verify you can:
  - [ ] Change "Chunking" dropdown (semantic, fixed, adaptive)
  - [ ] Change "Embedding" dropdown (small, large)
  - [ ] Change "Retrieval" dropdown (4 options)
  - [ ] Toggle "Reranking" checkbox
  - [ ] Change "LLM" dropdown (gpt-4o-mini, gpt-4o)
  - [ ] Change "Generation" dropdown (4 options)
- [ ] Hover over help icons and verify detailed tooltips appear
- [ ] Verify each strategy can have different settings

### Strategy Lab - Query Execution
**Note**: Requires ingested data and valid OPENAI_API_KEY

- [ ] Enter a test query in the text area
- [ ] Click "🚀 Run Comparison"
- [ ] Verify "Running strategies..." spinner appears
- [ ] Verify results appear in 3 columns
- [ ] Verify each result shows:
  - [ ] Strategy name
  - [ ] Execution time (⏱️)
  - [ ] Cost estimate (⚡/⚖️/🐌)
  - [ ] Generated answer
- [ ] Verify results look visually appealing with proper styling

### UI/UX Elements
- [ ] Verify all containers have proper borders and shadows
- [ ] Hover over strategy result cards and verify hover effect (shadow, slight lift)
- [ ] Verify emoji icons appear correctly throughout the UI
- [ ] Verify text is readable in both light and dark themes
- [ ] Verify spacing and padding looks good
- [ ] Verify buttons have hover effects
- [ ] Verify focus states work (tab through elements and verify green outline)

### Error Handling
- [ ] Remove .env file and restart app
- [ ] Verify error message appears in sidebar
- [ ] Verify labs show installation instructions
- [ ] Add .env back but remove OPENAI_API_KEY
- [ ] Verify warning appears about limited functionality
- [ ] Try to run ingestion without selecting files
- [ ] Verify warning message appears

## Known Limitations

1. **Full Pipeline Testing**: Requires a valid OpenAI API key to test ingestion and RAG strategies end-to-end
2. **Real-time Validation**: Some errors (like database connection issues) may only appear when actually running pipelines
3. **Performance**: Large documents or many documents may take significant time to process

## Next Steps for Full Testing

To fully test the ingestion and strategy pipelines:

1. Set up a valid OpenAI API key in `.env`
2. Upload 3-5 small test documents (PDFs, DOCX, or Markdown files)
3. Run ingestion with different chunking strategies
4. Test all retrieval methods in Strategy Lab
5. Compare results across different strategies
6. Test with complex multi-hop questions

## Reporting Issues

If you encounter any issues:
1. Note which test case failed
2. Check the browser console for JavaScript errors
3. Check the terminal running streamlit for Python errors
4. Take screenshots of the issue
5. Document the exact steps to reproduce
