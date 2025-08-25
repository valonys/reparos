"""
DigiTwin RAG Chatbot Module
A comprehensive RAG system with hybrid search, query rewriting, and streaming responses
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import requests
from datetime import datetime
import re

# Vector database imports
try:
    import weaviate
    from weaviate import Client
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

try:
    import faiss
    import faiss.cpu
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Embedding imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# LLM imports
try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Configuration
DB_PATH = 'notifs_data.db'
TABLE_NAME = 'notifications'
VECTOR_DB_PATH = 'vector_store'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

class DigiTwinRAG:
    """
    Comprehensive RAG system for DigiTwin notifications analysis
    """
    
    def __init__(self, db_path: str = DB_PATH, vector_db_path: str = VECTOR_DB_PATH):
        self.db_path = db_path
        self.vector_db_path = vector_db_path
        self.embedding_model = None
        self.vector_store = None
        self.llm_client = None
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all RAG components"""
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                st.success(f"✅ Embedding model loaded: {EMBEDDING_MODEL}")
            except Exception as e:
                st.error(f"❌ Failed to load embedding model: {e}")
        
        # Initialize vector store
        self.initialize_vector_store()
        
        # Initialize LLM clients
        self.initialize_llm_clients()
    
    def initialize_vector_store(self):
        """Initialize vector database (Weaviate or FAISS)"""
        if WEAVIATE_AVAILABLE:
            try:
                self.vector_store = Client("http://localhost:8080")
                st.success("✅ Weaviate vector store connected")
            except Exception as e:
                st.warning(f"⚠️ Weaviate not available: {e}")
                self.vector_store = None
        
        if not self.vector_store and FAISS_AVAILABLE:
            try:
                # Initialize FAISS index
                dimension = 384  # all-MiniLM-L6-v2 dimension
                self.vector_store = faiss.IndexFlatIP(dimension)
                st.success("✅ FAISS vector store initialized")
            except Exception as e:
                st.error(f"❌ Failed to initialize FAISS: {e}")
                self.vector_store = None
    
    def initialize_llm_clients(self):
        """Initialize LLM clients (Groq and Ollama)"""
        self.llm_client = {}
        
        # Initialize Groq client
        if GROQ_AVAILABLE:
            try:
                # You'll need to set GROQ_API_KEY in environment
                import os
                api_key = os.getenv('GROQ_API_KEY')
                if api_key:
                    self.llm_client['groq'] = groq.Groq(api_key=api_key)
                    st.success("✅ Groq client initialized")
                else:
                    st.warning("⚠️ GROQ_API_KEY not found in environment")
            except Exception as e:
                st.warning(f"⚠️ Groq initialization failed: {e}")
        
        # Initialize Ollama client
        if OLLAMA_AVAILABLE:
            try:
                self.llm_client['ollama'] = ollama.Client(host='http://localhost:11434')
                st.success("✅ Ollama client initialized")
            except Exception as e:
                st.warning(f"⚠️ Ollama initialization failed: {e}")
    
    def load_notifications_data(self) -> pd.DataFrame:
        """Load notifications data from SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(f'SELECT * FROM {TABLE_NAME}', conn)
            return df
        except Exception as e:
            st.error(f"❌ Failed to load data: {e}")
            return pd.DataFrame()
    
    def create_document_chunks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create document chunks for vectorization"""
        documents = []
        
        for idx, row in df.iterrows():
            # Create rich document representation
            doc = {
                'id': f"doc_{idx}",
                'content': f"""
                FPSO: {row.get('FPSO', 'N/A')}
                Notification Type: {row.get('Notifictn type', 'N/A')} 
                {'(Notification of Integrity)' if row.get('Notifictn type') == 'NI' else '(Notification of Conformity)' if row.get('Notifictn type') == 'NC' else ''}
                Description: {row.get('Description', 'N/A')}
                Created: {row.get('Created on', 'N/A')}
                Keywords: {row.get('Extracted_Keywords', 'N/A')}
                Modules: {row.get('Extracted_Modules', 'N/A')}
                Racks: {row.get('Extracted_Racks', 'N/A')}
                """.strip(),
                'metadata': {
                    'fpso': row.get('FPSO', 'N/A'),
                    'notification_type': row.get('Notifictn type', 'N/A'),
                    'created_date': row.get('Created on', 'N/A'),
                    'keywords': row.get('Extracted_Keywords', 'N/A'),
                    'modules': row.get('Extracted_Modules', 'N/A'),
                    'racks': row.get('Extracted_Racks', 'N/A')
                }
            }
            documents.append(doc)
        
        return documents
    
    def create_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Create embeddings for documents"""
        if not self.embedding_model:
            st.error("❌ Embedding model not available")
            return np.array([])
        
        texts = [doc['content'] for doc in documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def index_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Index documents in vector store"""
        if not self.vector_store:
            st.error("❌ Vector store not available")
            return
        
        if WEAVIATE_AVAILABLE and isinstance(self.vector_store, Client):
            # Index in Weaviate
            try:
                for doc, embedding in zip(documents, embeddings):
                    self.vector_store.data_object.create(
                        data_object=doc['metadata'],
                        class_name="Notification",
                        vector=embedding.tolist()
                    )
                st.success(f"✅ Indexed {len(documents)} documents in Weaviate")
            except Exception as e:
                st.error(f"❌ Failed to index in Weaviate: {e}")
        
        elif FAISS_AVAILABLE and hasattr(self.vector_store, 'add'):
            # Index in FAISS
            try:
                self.vector_store.add(embeddings.astype('float32'))
                # Save document metadata separately
                import pickle
                with open(f"{self.vector_db_path}_metadata.pkl", 'wb') as f:
                    pickle.dump(documents, f)
                st.success(f"✅ Indexed {len(documents)} documents in FAISS")
            except Exception as e:
                st.error(f"❌ Failed to index in FAISS: {e}")
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search (semantic + keyword)"""
        results = []
        
        # Semantic search
        if self.embedding_model and self.vector_store:
            query_embedding = self.embedding_model.encode([query])
            
            if WEAVIATE_AVAILABLE and isinstance(self.vector_store, Client):
                # Weaviate semantic search
                try:
                    semantic_results = self.vector_store.query.get("Notification", [
                        "fpso", "notification_type", "created_date", "keywords", "modules", "racks"
                    ]).with_near_vector({
                        "vector": query_embedding[0].tolist()
                    }).with_limit(k).do()
                    
                    for result in semantic_results['data']['Get']['Notification']:
                        results.append({
                            'content': f"FPSO: {result['fpso']}, Type: {result['notification_type']}, Keywords: {result['keywords']}",
                            'metadata': result,
                            'score': 1.0  # Weaviate doesn't return scores by default
                        })
                except Exception as e:
                    st.warning(f"⚠️ Weaviate search failed: {e}")
            
            elif FAISS_AVAILABLE and hasattr(self.vector_store, 'search'):
                # FAISS semantic search
                try:
                    scores, indices = self.vector_store.search(query_embedding.astype('float32'), k)
                    
                    # Load document metadata
                    import pickle
                    with open(f"{self.vector_db_path}_metadata.pkl", 'rb') as f:
                        documents = pickle.load(f)
                    
                    for score, idx in zip(scores[0], indices[0]):
                        if idx < len(documents):
                            results.append({
                                'content': documents[idx]['content'],
                                'metadata': documents[idx]['metadata'],
                                'score': float(score)
                            })
                except Exception as e:
                    st.warning(f"⚠️ FAISS search failed: {e}")
        
        # Keyword search as fallback
        if not results:
            df = self.load_notifications_data()
            if not df.empty:
                # Simple keyword matching
                query_terms = query.lower().split()
                for idx, row in df.iterrows():
                    text = f"{row.get('Description', '')} {row.get('Extracted_Keywords', '')}".lower()
                    if any(term in text for term in query_terms):
                        results.append({
                            'content': f"FPSO: {row.get('FPSO')}, Type: {row.get('Notifictn type')}, Description: {row.get('Description', '')[:100]}...",
                            'metadata': row.to_dict(),
                            'score': 0.5
                        })
                        if len(results) >= k:
                            break
        
        return results[:k]
    
    def query_rewriter(self, query: str) -> str:
        """Rewrite query for better retrieval"""
        rewrite_prompt = f"""
        Rewrite the following query to be more specific and searchable for FPSO notifications data.
        Focus on technical terms, FPSO names (GIR, DAL, PAZ, CLV), notification types (NI/NC), and equipment.
        
        Original query: {query}
        
        Rewritten query:"""
        
        # Use LLM to rewrite query
        rewritten_query = self.generate_response(rewrite_prompt, max_tokens=50, temperature=0.3)
        return rewritten_query.strip() if rewritten_query else query
    
    def generate_pivot_analysis(self, df: pd.DataFrame) -> str:
        """Generate pivot analysis summary"""
        analysis = []
        
        # FPSO distribution
        if 'FPSO' in df.columns:
            fpso_counts = df['FPSO'].value_counts()
            analysis.append(f"**FPSO Distribution:** {', '.join([f'{fpso}: {count}' for fpso, count in fpso_counts.items()])}")
        
        # Notification type distribution
        if 'Notifictn type' in df.columns:
            type_counts = df['Notifictn type'].value_counts()
            analysis.append(f"**Notification Types:** {', '.join([f'{ntype}: {count}' for ntype, count in type_counts.items()])}")
        
        # Keyword analysis
        if 'Extracted_Keywords' in df.columns:
            keywords = df['Extracted_Keywords'].str.split(', ').explode()
            keywords = keywords[keywords != 'None']
            if not keywords.empty:
                top_keywords = keywords.value_counts().head(5)
                analysis.append(f"**Top Keywords:** {', '.join([f'{kw}: {count}' for kw, count in top_keywords.items()])}")
        
        return "\n".join(analysis)
    
    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, stream: bool = False) -> str:
        """Generate response using available LLM"""
        
        # Try Groq first
        if 'groq' in self.llm_client:
            try:
                if stream:
                    return self._stream_groq_response(prompt, max_tokens, temperature)
                else:
                    response = self.llm_client['groq'].chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return response.choices[0].message.content
            except Exception as e:
                st.warning(f"⚠️ Groq generation failed: {e}")
        
        # Try Ollama as fallback
        if 'ollama' in self.llm_client:
            try:
                if stream:
                    return self._stream_ollama_response(prompt, max_tokens, temperature)
                else:
                    response = self.llm_client['ollama'].chat(
                        model='llama3.2',
                        messages=[{'role': 'user', 'content': prompt}]
                    )
                    return response['message']['content']
            except Exception as e:
                st.warning(f"⚠️ Ollama generation failed: {e}")
        
        return "I apologize, but I'm unable to generate a response at the moment. Please check your LLM configuration."
    
    def _stream_groq_response(self, prompt: str, max_tokens: int, temperature: float):
        """Stream response from Groq"""
        try:
            response = self.llm_client['groq'].chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            return full_response
        except Exception as e:
            st.error(f"❌ Groq streaming failed: {e}")
            return ""
    
    def _stream_ollama_response(self, prompt: str, max_tokens: int, temperature: float):
        """Stream response from Ollama"""
        try:
            response = self.llm_client['ollama'].chat(
                model='llama3.2',
                messages=[{'role': 'user', 'content': prompt}],
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    full_response += content
                    yield content
            
            return full_response
        except Exception as e:
            st.error(f"❌ Ollama streaming failed: {e}")
            return ""
    
    def create_rag_prompt(self, query: str, context: List[Dict[str, Any]], pivot_analysis: str) -> str:
        """Create optimized RAG prompt"""
        
        # Format context
        context_text = "\n\n".join([
            f"Document {i+1}:\n{doc['content']}\nRelevance Score: {doc['score']:.3f}"
            for i, doc in enumerate(context)
        ])
        
        prompt = f"""
        You are DigiTwin, an expert FPSO (Floating Production Storage and Offloading) notifications analyst.
        
        **Context Information:**
        {context_text}
        
        **Current Dataset Analysis:**
        {pivot_analysis}
        
        **Important Definitions:**
        - NI = Notification of Integrity (maintenance and safety notifications)
        - NC = Notification of Conformity (compliance and regulatory notifications)
        - FPSO Units: GIR, DAL, PAZ, CLV
        
        **User Query:** {query}
        
        Please provide a comprehensive, accurate response based on the context and dataset analysis. 
        Include specific details about FPSO units, notification types, and relevant insights.
        If the context doesn't contain enough information, say so clearly.
        
        **Response:**"""
        
        return prompt
    
    def process_query(self, query: str, stream: bool = True) -> str:
        """Process user query through the complete RAG pipeline"""
        
        # Step 1: Query rewriting
        rewritten_query = self.query_rewriter(query)
        
        # Step 2: Hybrid search
        search_results = self.hybrid_search(rewritten_query, k=5)
        
        # Step 3: Load data for pivot analysis
        df = self.load_notifications_data()
        pivot_analysis = self.generate_pivot_analysis(df) if not df.empty else "No data available"
        
        # Step 4: Create RAG prompt
        rag_prompt = self.create_rag_prompt(query, search_results, pivot_analysis)
        
        # Step 5: Generate response
        if stream:
            return self.generate_response(rag_prompt, max_tokens=800, temperature=0.7, stream=True)
        else:
            return self.generate_response(rag_prompt, max_tokens=800, temperature=0.7, stream=False)

def initialize_rag_system():
    """Initialize the RAG system"""
    with st.spinner("Initializing RAG system..."):
        rag = DigiTwinRAG()
        return rag

def render_chat_interface(rag: DigiTwinRAG):
    """Render the chat interface"""
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat header
    st.markdown("### 🤖 DigiTwin RAG Assistant")
    st.markdown("Ask me anything about your FPSO notifications data!")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your notifications data..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "avatar": "👤"
        })
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            
            try:
                # Process query with streaming
                full_response = ""
                for chunk in rag.process_query(prompt, stream=True):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "avatar": "🤖"
                })
                
            except Exception as e:
                error_msg = f"❌ Error processing query: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "avatar": "🤖"
                })
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🔧 RAG Controls")
        
        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Rebuild vector index
        if st.button("🔄 Rebuild Vector Index"):
            with st.spinner("Rebuilding vector index..."):
                df = rag.load_notifications_data()
                if not df.empty:
                    documents = rag.create_document_chunks(df)
                    embeddings = rag.create_embeddings(documents)
                    rag.index_documents(documents, embeddings)
                    st.success("✅ Vector index rebuilt!")
                else:
                    st.error("❌ No data available for indexing")

def main():
    """Main function to run the RAG chatbot"""
    st.set_page_config(page_title="DigiTwin RAG Assistant", layout="wide")
    
    # Initialize RAG system
    rag = initialize_rag_system()
    
    # Render chat interface
    render_chat_interface(rag)

if __name__ == "__main__":
    main()
