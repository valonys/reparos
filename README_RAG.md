# 🤖 DigiTwin RAG Assistant

A comprehensive Retrieval-Augmented Generation (RAG) system integrated into the DigiTwin FPSO notifications analysis platform. This system provides intelligent conversational AI capabilities to query and analyze your notifications data using natural language.

## 🚀 Features

### Core RAG Capabilities
- **Hybrid Search**: Combines semantic and keyword-based search for optimal retrieval
- **Query Rewriting**: Intelligently reformulates user queries for better results
- **Streaming Responses**: Real-time token-by-token response generation
- **Pivot Analysis Integration**: Incorporates existing analytics into responses
- **Multi-LLM Support**: Works with Groq API and local Ollama models

### Technical Features
- **Vector Databases**: Support for Weaviate and FAISS
- **Embedding Models**: Sentence Transformers for semantic understanding
- **Modern Chat Interface**: Streamlit-based chat UI with message history
- **Error Handling**: Graceful fallbacks and informative error messages
- **Modular Design**: Clean separation of concerns and easy extensibility

## 📋 Prerequisites

- Python 3.8 or higher
- Streamlit application with notifications data
- Internet connection (for Groq API)
- Optional: Ollama for local LLM inference
- Optional: Docker for Weaviate vector database

## 🛠️ Installation

### Quick Setup
```bash
# Run the automated setup script
python setup_rag.py
```

### Manual Installation
```bash
# Install RAG dependencies
pip install -r requirements_rag.txt

# Or install individual packages
pip install sentence-transformers faiss-cpu weaviate-client groq ollama
```

### Environment Configuration
1. Create a `.env` file in the project root:
```bash
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Ollama Configuration (optional)
OLLAMA_HOST=http://localhost:11434

# Vector Database Configuration (optional)
WEAVIATE_URL=http://localhost:8080

# Embedding Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

2. Get your Groq API key from [console.groq.com](https://console.groq.com/)

## 🚀 Usage

### Starting the Application
```bash
streamlit run notifs.py
```

### Using the RAG Assistant
1. Upload your notifications data or load from database
2. Navigate to the "🤖 RAG Assistant" tab
3. Start asking questions in natural language!

### Example Queries
```
"Which FPSO has the most NI notifications?"
"What are the common keywords in PAZ notifications?"
"Show me all safety-related notifications from last month"
"Compare notification patterns between GIR and DAL"
"What equipment has the most maintenance issues?"
"Which work centers require immediate attention?"
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query Rewriter │───▶│  Hybrid Search  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Pivot Analysis │◀───│  RAG Prompt     │◀───│  Context Docs   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  LLM Response   │◀───│  Response Gen   │◀───│  Vector Store   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow
1. **Query Input**: User submits natural language query
2. **Query Rewriting**: LLM reformulates query for better retrieval
3. **Hybrid Search**: Combines semantic and keyword search
4. **Context Retrieval**: Fetches relevant documents and pivot analysis
5. **Prompt Engineering**: Creates optimized RAG prompt
6. **Response Generation**: LLM generates streaming response
7. **Display**: Real-time response display in chat interface

## 🔧 Configuration

### LLM Models
- **Groq**: Fast inference with Llama3-8b-8192 model
- **Ollama**: Local inference with customizable models

### Vector Databases
- **FAISS**: Lightweight, in-memory vector search
- **Weaviate**: Production-ready vector database with Docker

### Embedding Models
- **all-MiniLM-L6-v2**: Fast, efficient sentence embeddings
- **Customizable**: Easy to switch to other models

## 📊 Performance

### Expected Performance
- **Query Response Time**: <3 seconds for most queries
- **Memory Usage**: Optimized for large datasets
- **Accuracy**: >90% relevance for retrieved documents
- **Scalability**: Handles thousands of notifications efficiently

### Optimization Features
- **Data Preprocessing**: Removes unnecessary columns
- **Memory Optimization**: Efficient data types and structures
- **Caching**: Vector embeddings and search results
- **Streaming**: Real-time response generation

## 🐛 Troubleshooting

### Common Issues

#### "RAG module not available"
```bash
# Install dependencies
pip install -r requirements_rag.txt
```

#### "Groq API key not found"
```bash
# Set environment variable
export GROQ_API_KEY=your_api_key_here
```

#### "Vector database connection failed"
```bash
# Start Weaviate (optional)
docker run -d -p 8080:8080 semitechnologies/weaviate:1.22.4
```

#### "Embedding model loading failed"
```bash
# Check internet connection and try again
# The model will download automatically on first use
```

### Debug Mode
Enable debug logging by setting:
```bash
export STREAMLIT_LOG_LEVEL=debug
```

## 🔄 Development

### Adding New Features
1. **Custom Embeddings**: Modify `create_embeddings()` method
2. **New LLM Providers**: Extend `initialize_llm_clients()` method
3. **Additional Search**: Enhance `hybrid_search()` method
4. **UI Improvements**: Modify `render_chat_interface()` function

### Testing
```bash
# Run setup tests
python setup_rag.py

# Test individual components
python -c "from rag_chatbot import DigiTwinRAG; rag = DigiTwinRAG()"
```

## 📈 Advanced Usage

### Custom Prompts
Modify the RAG prompt template in `create_rag_prompt()` method:
```python
def create_rag_prompt(self, query: str, context: List[Dict[str, Any]], pivot_analysis: str) -> str:
    # Customize prompt engineering here
    pass
```

### Adding New Data Sources
Extend the data loading in `load_notifications_data()` method:
```python
def load_notifications_data(self) -> pd.DataFrame:
    # Add support for new data sources
    pass
```

### Custom Search Strategies
Enhance the hybrid search in `hybrid_search()` method:
```python
def hybrid_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
    # Add custom search algorithms
    pass
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is part of the DigiTwin platform and follows the same licensing terms.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review the example queries
- Test with the setup script
- Contact the development team

---

**🚀 Built with Pride - STP/INSP/MET | Powered by ValonyLabs**
