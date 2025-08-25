#!/usr/bin/env python3
"""
DigiTwin RAG Setup Script
Helps users install and configure the RAG system dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install RAG dependencies"""
    print("🚀 Installing DigiTwin RAG Dependencies")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install core dependencies
    dependencies = [
        ("sentence-transformers", "Sentence Transformers for embeddings"),
        ("faiss-cpu", "FAISS vector database"),
        ("weaviate-client", "Weaviate vector database client"),
        ("groq", "Groq LLM API client"),
        ("ollama", "Ollama local LLM client"),
        ("numpy", "Numerical computing"),
        ("pandas", "Data manipulation"),
        ("streamlit", "Web application framework")
    ]
    
    success_count = 0
    for package, description in dependencies:
        if run_command(f"pip install {package}", f"Installing {description}"):
            success_count += 1
    
    print(f"\n📊 Installation Summary: {success_count}/{len(dependencies)} packages installed successfully")
    return success_count == len(dependencies)

def setup_environment():
    """Setup environment variables and configuration"""
    print("\n🔧 Setting up environment...")
    
    # Create .env file template
    env_content = """# DigiTwin RAG Environment Configuration

# Groq API Configuration
# Get your API key from: https://console.groq.com/
GROQ_API_KEY=your_groq_api_key_here

# Ollama Configuration (optional)
# Install Ollama from: https://ollama.ai/
OLLAMA_HOST=http://localhost:11434

# Vector Database Configuration
# Weaviate (optional) - Install with: docker run -d -p 8080:8080 semitechnologies/weaviate:1.22.4
WEAVIATE_URL=http://localhost:8080

# Embedding Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("📝 Please edit .env file with your API keys")
    else:
        print("ℹ️ .env file already exists")

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "vector_store",
        "logs",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_installation():
    """Test the RAG installation"""
    print("\n🧪 Testing RAG installation...")
    
    test_script = """
import sys
import importlib

# Test imports
modules_to_test = [
    'sentence_transformers',
    'faiss',
    'weaviate',
    'groq',
    'ollama',
    'numpy',
    'pandas',
    'streamlit'
]

print("Testing module imports...")
for module in modules_to_test:
    try:
        importlib.import_module(module)
        print(f"✅ {module}")
    except ImportError as e:
        print(f"❌ {module}: {e}")

print("\\nTesting embedding model...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    test_embedding = model.encode(['test sentence'])
    print(f"✅ Embedding model working (shape: {test_embedding.shape})")
except Exception as e:
    print(f"❌ Embedding model failed: {e}")

print("\\nRAG system test completed!")
"""
    
    with open("test_rag.py", "w") as f:
        f.write(test_script)
    
    if run_command("python test_rag.py", "Running RAG system test"):
        print("✅ RAG system test passed!")
        os.remove("test_rag.py")
    else:
        print("❌ RAG system test failed. Please check the errors above.")

def main():
    """Main setup function"""
    print("🤖 DigiTwin RAG Setup")
    print("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Some dependencies failed to install. Please check the errors above.")
        return
    
    # Setup environment
    setup_environment()
    
    # Create directories
    create_directories()
    
    # Test installation
    test_installation()
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Install Ollama (optional): https://ollama.ai/")
    print("3. Start Weaviate (optional): docker run -d -p 8080:8080 semitechnologies/weaviate:1.22.4")
    print("4. Run the application: streamlit run notifs.py")
    print("\n🚀 Happy coding with DigiTwin RAG!")

if __name__ == "__main__":
    main()
