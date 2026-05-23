import streamlit as st
import os
from PyPDF2 import PdfReader
import cassio
from langchain_community.vectorstores import Cassandra
from langchain_classic.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables if available
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF RAG Assistant with Astra DB & Groq",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600;700&display=swap');
        
        /* Font styles */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
            padding: 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        .main-header h1 {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 2.8rem !important;
            margin: 0;
            letter-spacing: -0.025em;
        }
        
        .main-header p {
            margin: 0.75rem 0 0 0;
            opacity: 0.95;
            font-size: 1.15rem;
            font-weight: 300;
        }
        
        /* Custom card style */
        .premium-card {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .premium-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            border-color: rgba(99, 102, 241, 0.4);
        }
        
        /* Chat bubble styles */
        .chat-bubble {
            padding: 1.25rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            line-height: 1.6;
        }
        .chat-user {
            background-color: rgba(99, 102, 241, 0.15);
            border-left: 5px solid #6366f1;
        }
        .chat-assistant {
            background-color: rgba(6, 182, 212, 0.1);
            border-left: 5px solid #06b6d4;
        }
        
        /* Highlight labels */
        .doc-tag {
            background-color: rgba(99, 102, 241, 0.1);
            color: #818cf8;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.85rem;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Main Hero Header
st.markdown("""
    <div class="main-header">
        <h1>📄 AstraDB & Groq RAG Engine</h1>
        <p>Upload PDFs, index them in Cassandra vector search, and query with high-speed Groq LLMs</p>
    </div>
""", unsafe_allow_html=True)

# Initialize Session States
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Configuration
st.sidebar.markdown("### 🔑 API & Database Secrets")

# DB Credentials Setup (Auto-populate from environment variables if present)
astra_db_id_env = os.environ.get("ASTRA_DB_ID", "")
astra_db_token_env = os.environ.get("ASTRA_DB_APPLICATION_TOKEN", "")
groq_key_env = os.environ.get("GROQ_API_KEY", "")
openai_key_env = os.environ.get("OPENAI_API_KEY", "")
google_key_env = os.environ.get("GOOGLE_API_KEY", "")

astra_db_id = st.sidebar.text_input(
    "Astra DB ID",
    value=astra_db_id_env,
    type="password" if astra_db_id_env else "default",
    placeholder="e.g. 56eada22-55b6-4100..."
)

astra_db_token = st.sidebar.text_input(
    "Astra DB Application Token",
    value=astra_db_token_env,
    type="password",
    placeholder="e.g. AstraCS:..."
)

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    value=groq_key_env,
    type="password",
    placeholder="gsk_..."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Configuration Settings")

# Model configuration
groq_model = st.sidebar.text_input(
    "Groq Model Name / ID",
    value="llama-3.1-8b-instant",
    placeholder="e.g. llama-3.1-8b-instant, llama-3.3-70b-versatile..."
).strip()

# Embedding Provider Selection
embedding_provider = st.sidebar.selectbox(
    "Embedding Provider",
    ["HuggingFace (Local, Free)", "OpenAI (Requires Key)", "Google Gemini (Requires Key)"],
    index=0
)

# Render keys based on choice
embedding_api_key = ""
if embedding_provider == "OpenAI (Requires Key)":
    embedding_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=openai_key_env,
        type="password"
    )
elif embedding_provider == "Google Gemini (Requires Key)":
    embedding_api_key = st.sidebar.text_input(
        "Google API Key",
        value=google_key_env,
        type="password"
    )

astra_table_name = st.sidebar.text_input(
    "Astra DB Vector Table Name",
    value="qa_mini_demo"
)

# Expandable Settings for Chunking
with st.sidebar.expander("📝 Chunking Parameters", expanded=False):
    chunk_size = st.number_input("Chunk Size", value=800, min_value=100, step=50)
    chunk_overlap = st.number_input("Chunk Overlap", value=200, min_value=0, step=25)

# Validate Sidebar Configurations
def validate_inputs():
    if not astra_db_id or not astra_db_token:
        st.error("⚠️ Please provide both Astra DB ID and Application Token.")
        return False
    if not groq_api_key:
        st.error("⚠️ Please provide a Groq API Key.")
        return False
    if embedding_provider == "OpenAI (Requires Key)" and not embedding_api_key:
        st.error("⚠️ OpenAI Embedding selected, but no OpenAI API Key provided.")
        return False
    if embedding_provider == "Google Gemini (Requires Key)" and not embedding_api_key:
        st.error("⚠️ Google Gemini Embedding selected, but no Google API Key provided.")
        return False
    return True

# Lazy imports to speed up loading
def get_embeddings():
    if embedding_provider == "HuggingFace (Local, Free)":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        # Load local model
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    elif embedding_provider == "OpenAI (Requires Key)":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(openai_api_key=embedding_api_key)
    elif embedding_provider == "Google Gemini (Requires Key)":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(google_api_key=embedding_api_key, model="models/embedding-001")
    return None

def get_groq_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=groq_model,
        temperature=0.2
    )

# App Content Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📥 1. Upload & Index PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file to index",
        type="pdf",
        help="Upload a document to extract text and build vector representations in Astra DB."
    )
    
    if uploaded_file is not None:
        st.info(f"📁 Selected File: **{uploaded_file.name}** ({len(uploaded_file.getvalue()) / 1024:.1f} KB)")
        
        # Action button to trigger vectorization
        if st.button("🚀 Process & Upload to Astra DB", use_container_width=True):
            if validate_inputs():
                try:
                    with st.spinner("Extracting text from PDF..."):
                        # Read PDF
                        pdfreader = PdfReader(uploaded_file)
                        raw_text = ''
                        for i, page in enumerate(pdfreader.pages):
                            content = page.extract_text()
                            if content:
                                raw_text += content
                                
                        if not raw_text.strip():
                            st.error("❌ No text could be extracted from the uploaded PDF. Please make sure the PDF contains readable text.")
                        else:
                            st.success(f"✅ Extracted {len(raw_text)} characters of raw text.")
                            
                            # Split Text
                            with st.spinner("Splitting text into chunks..."):
                                text_splitter = CharacterTextSplitter(
                                    separator="\n",
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap,
                                    length_function=len,
                                )
                                texts = text_splitter.split_text(raw_text)
                                st.success(f"✅ Split text into {len(texts)} chunks.")
                            
                            # Initialize DB Connection
                            with st.spinner("Connecting to Astra DB via CassIO..."):
                                cassio.init(token=astra_db_token, database_id=astra_db_id)
                                
                            # Initialize Embeddings
                            with st.spinner(f"Initializing {embedding_provider} Model..."):
                                embeddings = get_embeddings()
                                
                            # Initialize Vector Store
                            with st.spinner("Preparing Cassandra Vector Store..."):
                                astra_vector_store = Cassandra(
                                    embedding=embeddings,
                                    table_name=astra_table_name,
                                    session=None,
                                    keyspace=None,
                                )
                                
                            # Load texts in batches with progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            batch_size = 20
                            total_chunks = len(texts)
                            
                            for idx in range(0, total_chunks, batch_size):
                                batch = texts[idx:idx + batch_size]
                                status_text.write(f"Uploading chunks {idx+1} to {min(idx + batch_size, total_chunks)} of {total_chunks}...")
                                astra_vector_store.add_texts(batch)
                                
                                # Update progress
                                progress = min((idx + batch_size) / total_chunks, 1.0)
                                progress_bar.progress(progress)
                                
                            status_text.empty()
                            progress_bar.empty()
                            
                            # Build index wrapper
                            astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
                            
                            # Store in session state
                            st.session_state.vector_store = astra_vector_store
                            st.session_state.vector_index = astra_vector_index
                            st.session_state.indexed = True
                            
                            st.balloons()
                            st.success(f"🎉 Successfully inserted {total_chunks} chunks into table '{astra_table_name}'!")
                            
                except Exception as e:
                    st.error(f"❌ Error during vectorization: {str(e)}")
                    st.exception(e)

with col2:
    st.markdown("### 💬 2. Ask the Document")
    
    if not st.session_state.indexed:
        st.warning("👈 Please set up your secrets, upload a PDF, and run the indexer first!")
    else:
        st.info("💡 PDF Indexed! Enter your question below to query your document.")
        
        # Pre-baked questions support based on standard PDF speaches
        st.markdown("**Suggested Questions:**")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            q1 = st.button("What are the key highlights?", use_container_width=True)
        with s_col2:
            q2 = st.button("What is the financial breakdown?", use_container_width=True)
            
        # Chat interface
        query_text = st.chat_input("Enter your question...")
        
        # Trigger query if suggestion is clicked or input is provided
        selected_query = ""
        if q1:
            selected_query = "What are the key highlights of the speech/document?"
        elif q2:
            selected_query = "What is the financial breakdown or budget allocations mentioned?"
        elif query_text:
            selected_query = query_text
            
        if selected_query:
            if validate_inputs():
                try:
                    with st.spinner("Thinking..."):
                        # Get LLM and wrapper
                        llm = get_groq_llm()
                        astra_vector_index = st.session_state.vector_index
                        astra_vector_store = st.session_state.vector_store
                        
                        # Run RAG Query
                        answer = astra_vector_index.query(selected_query, llm=llm).strip()
                        
                        # Add to chat history
                        st.session_state.chat_history.append((selected_query, answer))
                        
                        # Perform similarity search with score to get source documentation
                        sources = astra_vector_store.similarity_search_with_score(selected_query, k=3)
                        st.session_state.sources = sources
                        
                except Exception as e:
                    st.error(f"❌ Query Error: {str(e)}")
                    st.exception(e)
                    
        # Render Chat History (Newest first or chronological - let's do chronological)
        if st.session_state.chat_history:
            for q, a in st.session_state.chat_history:
                st.markdown(f"""
                    <div class="chat-bubble chat-user">
                        <strong>User:</strong><br>{q}
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class="chat-bubble chat-assistant">
                        <strong>Assistant (Groq - {groq_model}):</strong><br>{a}
                    </div>
                """, unsafe_allow_html=True)
                
            # Render Sources for the most recent query
            if hasattr(st.session_state, "sources") and st.session_state.sources:
                with st.expander("🔍 RAG Explorer: Retrieved Context & Relevance", expanded=True):
                    st.markdown("Here are the top retrieved text segments from Astra DB based on vector similarity:")
                    for idx, (doc, score) in enumerate(st.session_state.sources):
                        st.markdown(f"""
                            <div class="premium-card">
                                <div>
                                    <span class="doc-tag">Document Chunk #{idx+1}</span> 
                                    <span class="doc-tag" style="background-color: rgba(6, 182, 212, 0.1); color: #06b6d4;">Similarity Score: {score:.4f}</span>
                                </div>
                                <p style="margin-top:0.8rem; font-style: italic; font-size:0.95rem;">
                                    "... {doc.page_content} ..."
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
            # Button to clear history
            if st.button("🧹 Clear Chat History"):
                st.session_state.chat_history = []
                if hasattr(st.session_state, "sources"):
                    del st.session_state.sources
                st.rerun()
