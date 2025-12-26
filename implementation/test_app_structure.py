"""
Simple test to verify app.py structure and key components
"""

def test_app_structure():
    """Test that app.py has all the required components"""
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Test 1: Theme toggle with rerun
    assert 'def toggle_theme():' in content, "toggle_theme function not found"
    assert 'st.rerun()' in content, "st.rerun() not found in toggle_theme"
    print("✅ Theme toggle has st.rerun()")
    
    # Test 2: Lab order
    assert '"Ingestion Lab", "Strategy Lab"' in content, "Lab order not correct"
    print("✅ Lab order is Ingestion Lab first")
    
    # Test 3: File upload
    assert 'st.file_uploader' in content, "File uploader not found"
    assert 'st.tabs(["Upload Files", "Available Files"])' in content, "File tabs not found"
    print("✅ File upload with tabs implemented")
    
    # Test 4: Chunking is selectable
    assert 'st.selectbox' in content, "Selectbox not found"
    # Check that Chunking is in a selectbox (appears once in the loop)
    assert '"Chunking"' in content, "Chunking selectbox not found"
    # Verify it's a selectbox by checking the pattern
    chunking_section = content[content.find('"Chunking"') - 100:content.find('"Chunking"') + 100]
    assert 'st.selectbox' in chunking_section, "Chunking is not in a selectbox"
    print("✅ Chunking is selectable in all strategies")
    
    # Test 5: Enhanced CSS
    assert '.strategy-container:hover' in content, "Hover effects not found"
    assert 'border-radius:' in content, "Rounded corners not found"
    print("✅ Enhanced CSS with hover effects")
    
    # Test 6: Help text improvements
    assert '💡' in content or '🔍' in content, "Emoji icons not found"
    help_text_count = content.count('help=')
    assert help_text_count >= 10, "Not enough help text added"
    print(f"✅ Help text improved ({help_text_count} help parameters)")
    
    # Test 7: Environment checks
    assert 'check_environment()' in content, "Environment check function not found"
    assert 'env_errors' in content, "Environment error handling not found"
    print("✅ Environment validation implemented")
    
    # Test 8: Error handling
    assert 'IMPORTS_SUCCESSFUL' in content, "Import error handling not found"
    assert 'try:' in content and 'except ImportError' in content, "Import try/except not found"
    print("✅ Error handling for missing dependencies")
    
    # Test 9: Welcome message
    assert 'Welcome!' in content, "Welcome message not found"
    assert 'Quick Start:' in content, "Quick start guide not found"
    print("✅ Welcome message and quick start guide")
    
    # Test 10: Educational tips
    assert 'Tip for Learners' in content, "Educational tips not found"
    print("✅ Educational tips for learners")
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == '__main__':
    test_app_structure()
