import os
import time
import json
import re
import math
import streamlit as st
import numpy as np
import pypdf
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

# Load local environment variables (if any)
load_dotenv()

# Set page configuration with a modern theme
st.set_page_config(
    page_title="Multimodal PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Apply custom font */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Gradient Title */
.gradient-text {
    background: linear-gradient(135deg, #4F46E5 0%, #EC4899 50%, #F59E0B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    text-align: center;
}

.subtitle-text {
    color: #6B7280;
    text-align: center;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Premium Card Design for Stats */
.stat-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.25rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-4px);
    border-color: rgba(79, 70, 229, 0.4);
    box-shadow: 0 10px 25px rgba(79, 70, 229, 0.1);
}

.stat-val {
    font-size: 2rem;
    font-weight: 700;
    color: #4F46E5;
    margin-bottom: 0.25rem;
}

.stat-lbl {
    font-size: 0.85rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* Citation Source Bubble */
.citation-container {
    background: rgba(255, 255, 255, 0.02);
    border-left: 3px solid #EC4899;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
    border-radius: 0 12px 12px 0;
    font-size: 0.9rem;
}

.citation-header {
    font-weight: 600;
    color: #EC4899;
    margin-bottom: 0.25rem;
}

.citation-snippet {
    font-style: italic;
    color: #9CA3AF;
}

/* Glassmorphic elements for Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0B0F19 !important;
}

/* Status success style */
.key-valid {
    color: #10B981;
    font-weight: 600;
    font-size: 0.85rem;
}

.key-invalid {
    color: #EF4444;
    font-weight: 600;
    font-size: 0.85rem;
}

/* Welcome Card */
.welcome-card {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%);
    border: 1px solid rgba(79, 70, 229, 0.2);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 2rem auto;
    max-width: 800px;
}
</style>
""", unsafe_allow_html=True)

# TF-IDF Keyword Indexing Functions
def tokenize(text):
    return re.findall(r'\w+', text.lower())

def index_tfidf(chunks):
    """Computes TF-IDF vectors for all chunks in pure Python/NumPy."""
    chunk_tokens = [tokenize(c["text"]) for c in chunks]
    
    vocab = set()
    for tokens in chunk_tokens:
        vocab.update(tokens)
    vocab = list(vocab)
    vocab_idx = {word: idx for idx, word in enumerate(vocab)}
    
    df = {word: 0 for word in vocab}
    for tokens in chunk_tokens:
        unique_tokens = set(tokens)
        for t in unique_tokens:
            df[t] += 1
            
    num_docs = len(chunks)
    idf = {}
    for word in vocab:
        idf[word] = math.log((1 + num_docs) / (1 + df[word])) + 1.0
        
    vectors = np.zeros((num_docs, len(vocab)), dtype=np.float32)
    for doc_idx, tokens in enumerate(chunk_tokens):
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t, freq in tf.items():
            vectors[doc_idx, vocab_idx[t]] = freq * idf[t]
            
    return {
        "vocab_idx": vocab_idx,
        "idf": idf,
        "vectors": vectors
    }

def query_tfidf(query, tfidf_index):
    """Computes cosine similarity of query tokens against document TF-IDF vectors."""
    vocab_idx = tfidf_index["vocab_idx"]
    idf = tfidf_index["idf"]
    vectors = tfidf_index["vectors"]
    num_docs = vectors.shape[0]
    
    query_tokens = tokenize(query)
    query_vector = np.zeros(len(vocab_idx), dtype=np.float32)
    
    q_tf = {}
    for t in query_tokens:
        q_tf[t] = q_tf.get(t, 0) + 1
        
    for t, freq in q_tf.items():
        if t in vocab_idx:
            query_vector[vocab_idx[t]] = freq * idf[t]
            
    q_norm_val = np.linalg.norm(query_vector)
    if q_norm_val == 0:
        return np.zeros(num_docs)
        
    q_norm = query_vector / q_norm_val
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_db = vectors / norms
    
    return np.dot(normalized_db, q_norm)

# Core Helper Functions
def extract_and_chunk_pdfs(uploaded_files, chunk_size, chunk_overlap):
    """Extracts text page-by-page from multiple PDFs and generates semantic chunks."""
    all_chunks = []
    doc_stats = {}
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        uploaded_file.seek(0)
        
        try:
            reader = pypdf.PdfReader(uploaded_file)
            pages_data = []
            total_words = 0
            
            for idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_data.append({
                        "page_num": idx + 1,
                        "text": text
                    })
                    total_words += len(text.split())
            
            doc_chunks = []
            for page in pages_data:
                page_num = page["page_num"]
                text = page["text"]
                
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    chunk_content = text[start:end].strip()
                    
                    if len(chunk_content) > 0:
                        doc_chunks.append({
                            "text": chunk_content,
                            "page_num": page_num,
                            "source": filename
                        })
                    
                    if len(text) <= end:
                        break
                    start += (chunk_size - chunk_overlap)
            
            all_chunks.extend(doc_chunks)
            doc_stats[filename] = {
                "pages": len(reader.pages),
                "chunks": len(doc_chunks),
                "words": total_words
            }
        except Exception as e:
            st.error(f"Error reading file '{filename}': {str(e)}")
            
    return all_chunks, doc_stats

def get_embeddings(texts, api_key):
    """Generates embeddings using Gemini API (text-embedding-004) in batches."""
    genai.configure(api_key=api_key)
    embeddings = []
    batch_size = 50
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=batch
        )
        embeddings.extend(response['embedding'])
        
    return np.array(embeddings)

def validate_gemini_key(api_key):
    """Validates Gemini API Key by generating a test embedding."""
    if not api_key:
        return False
    try:
        genai.configure(api_key=api_key)
        genai.embed_content(model="models/text-embedding-004", content="test")
        return True
    except Exception:
        return False

def validate_groq_key(api_key):
    """Validates Groq API Key by sending a tiny test chat message."""
    if not api_key:
        return False
    try:
        client = Groq(api_key=api_key)
        client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=2
        )
        return True
    except Exception:
        return False

def generate_response(provider, model_name, api_key, system_prompt, prompt_text, temperature):
    """Unified text generation function routing to Gemini or Groq API."""
    if provider == "Google Gemini":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
        response = model.generate_content(
            prompt_text,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature
            )
        )
        return response.text
    elif provider == "Groq Cloud":
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}
if "last_upload_hash" not in st.session_state:
    st.session_state.last_upload_hash = ""
if "last_embed_engine" not in st.session_state:
    st.session_state.last_embed_engine = ""
if "last_provider" not in st.session_state:
    st.session_state.last_provider = ""

# Sidebar Section
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: white;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # AI Provider Select
    st.markdown("### 1. Provider & API Configuration")
    provider = st.selectbox(
        "Select AI Provider",
        ["Google Gemini", "Groq Cloud"],
        index=1, # Default to Groq Cloud as requested by user
        help="Choose whether to use Google Gemini or Groq Cloud."
    )
    
    # Configure API Keys
    active_api_key = ""
    api_key_valid = False
    
    if provider == "Google Gemini":
        env_gemini_key = os.getenv("GEMINI_API_KEY", "")
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=env_gemini_key,
            help="Get your key from Google AI Studio."
        )
        active_api_key = gemini_key
        if gemini_key:
            with st.spinner("Validating Gemini key..."):
                api_key_valid = validate_gemini_key(gemini_key)
                if api_key_valid:
                    st.markdown("<span class='key-valid'>✓ Gemini API Key Validated</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='key-invalid'>✗ Invalid Gemini API Key</span>", unsafe_allow_html=True)
        else:
            st.info("💡 Please enter your Gemini API Key.")
            
    elif provider == "Groq Cloud":
        env_groq_key = os.getenv("GROQ_API_KEY", "")
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value=env_groq_key,
            help="Get your key from Groq Console."
        )
        active_api_key = groq_key
        if groq_key:
            with st.spinner("Validating Groq key..."):
                api_key_valid = validate_groq_key(groq_key)
                if api_key_valid:
                    st.markdown("<span class='key-valid'>✓ Groq API Key Validated</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='key-invalid'>✗ Invalid Groq API Key</span>", unsafe_allow_html=True)
        else:
            st.info("💡 Please enter your Groq API Key.")
            
    st.markdown("---")
    
    # Document Upload
    st.markdown("### 2. Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents to chat with."
    )
    
    # Check if files changed
    upload_hash = "".join([f"{f.name}-{f.size}" for f in uploaded_files]) if uploaded_files else ""
    
    st.markdown("---")
    
    # Hyperparameters
    st.markdown("### 3. Hyperparameters")
    
    # Engine logic
    if provider == "Google Gemini":
        embed_engine = "Gemini Embeddings"
        model_options = ["gemini-2.5-flash", "gemini-2.5-pro"]
    else:
        # Groq Cloud
        embed_engine = st.selectbox(
            "Embedding / Search Engine",
            ["TF-IDF (Local, Keyless)", "Gemini Embeddings (Requires Gemini Key)"],
            index=0,
            help="TF-IDF is completely local and keyless. Gemini embeddings offer superior semantic context but require a Gemini API Key."
        )
        model_options = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
        
    selected_model = st.selectbox("LLM Model", model_options, index=0)
    
    # Embedding Configuration dependencies
    gemini_key_for_embed = ""
    if provider == "Groq Cloud" and embed_engine == "Gemini Embeddings (Requires Gemini Key)":
        gemini_key_for_embed = st.text_input(
            "Gemini Key (for Embeddings)",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Needed only to generate semantic vector embeddings."
        )
        if not validate_gemini_key(gemini_key_for_embed):
            st.markdown("<span class='key-invalid'>✗ Invalid Gemini Embedding Key</span>", unsafe_allow_html=True)
            
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
    chunk_size = st.number_input("Chunk Size (Chars)", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap (Chars)", min_value=0, max_value=1000, value=200, step=50)
    top_k = st.slider("Retrieve Top Chunks (K)", min_value=2, max_value=10, value=5)
    
    st.markdown("---")
    
    # Reset Application Button
    if st.button("🗑️ Clear Cache & Conversations", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.processed_files = []
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.doc_stats = {}
        st.session_state.last_upload_hash = ""
        st.session_state.last_embed_engine = ""
        st.session_state.last_provider = ""
        st.rerun()

# Determine if we can proceed with document indexing
can_index = False
if api_key_valid and uploaded_files:
    if provider == "Google Gemini":
        can_index = True
    elif provider == "Groq Cloud":
        if embed_engine == "TF-IDF (Local, Keyless)":
            can_index = True
        elif embed_engine == "Gemini Embeddings (Requires Gemini Key)" and validate_gemini_key(gemini_key_for_embed):
            can_index = True

# Process uploaded files (re-trigger if configs change)
config_changed = (
    (upload_hash != st.session_state.last_upload_hash) or
    (embed_engine != st.session_state.last_embed_engine) or
    (provider != st.session_state.last_provider)
)

if can_index and config_changed:
    st.session_state.last_upload_hash = upload_hash
    st.session_state.last_embed_engine = embed_engine
    st.session_state.last_provider = provider
    
    with st.status("🔮 Processing your PDF documents...", expanded=True) as status:
        status.update(label="Parsing PDF documents and extracting text...", state="running")
        chunks, doc_stats = extract_and_chunk_pdfs(uploaded_files, chunk_size, chunk_overlap)
        
        if chunks:
            if provider == "Google Gemini" or (provider == "Groq Cloud" and embed_engine.startswith("Gemini")):
                status.update(label="Computing semantic vector embeddings using Gemini API...", state="running")
                key_to_use = active_api_key if provider == "Google Gemini" else gemini_key_for_embed
                embeddings = get_embeddings([c["text"] for c in chunks], key_to_use)
                st.session_state.embeddings = embeddings
            else:
                # TF-IDF fallback
                status.update(label="Compiling local TF-IDF vocabulary index...", state="running")
                tfidf_index = index_tfidf(chunks)
                st.session_state.embeddings = tfidf_index
                
            # Save stats & chunks
            st.session_state.chunks = chunks
            st.session_state.doc_stats = doc_stats
            st.session_state.processed_files = [f.name for f in uploaded_files]
            
            status.update(label=f"✨ Document Vector Index Ready ({embed_engine})!", state="complete")
            st.toast("Success: Documents indexed successfully!", icon="✅")
        else:
            status.update(label="⚠️ No readable text found in PDFs.", state="error")

# Main Page Interface
st.markdown("<h1 class='gradient-text'>Multi-Provider PDF Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Interrogate your documents with lightning-fast speeds powered by Groq or Gemini.</p>", unsafe_allow_html=True)

# 1. Stats Dashboard
if st.session_state.processed_files:
    total_docs = len(st.session_state.processed_files)
    total_pages = sum([stats["pages"] for stats in st.session_state.doc_stats.values()])
    total_chunks = len(st.session_state.chunks)
    total_words = sum([stats["words"] for stats in st.session_state.doc_stats.values()])
    
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-val">{total_docs}</div>
            <div class="stat-lbl">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-val">{total_pages}</div>
            <div class="stat-lbl">Total Pages</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-val">{total_chunks}</div>
            <div class="stat-lbl">Index Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-val">{total_words:,}</div>
            <div class="stat-lbl">Total Words</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# 2. Main Chat Area
if not api_key_valid:
    st.markdown(f"""
    <div class="welcome-card">
        <h3>🔑 {provider} API Key Required</h3>
        <p style="color: #6B7280; font-size: 1.05rem; margin-top: 0.5rem;">
            Please enter your valid API key in the sidebar panel to unlock chat features.
        </p>
    </div>
    """, unsafe_allow_html=True)
elif not st.session_state.processed_files:
    st.markdown("""
    <div class="welcome-card">
        <h3>📂 Upload PDF Documents</h3>
        <p style="color: #6B7280; font-size: 1.05rem; margin-top: 0.5rem;">
            Upload one or more PDF files in the sidebar panel.<br>
            Once processed, you can search and chat with them instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Render Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 View Referenced Sources"):
                    for idx, src in enumerate(message["sources"]):
                        st.markdown(f"""
                        <div class="citation-container">
                            <div class="citation-header">Source {idx+1}: {src['source']} (Page {src['page_num']}) • Match Score: {src['score']:.1f}%</div>
                            <div class="citation-snippet">"... {src['text']} ..."</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input Box
    if user_query := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant"):
            with st.spinner("Searching document index..."):
                try:
                    # 1. Similarity Retrieval
                    db_index = st.session_state.embeddings
                    
                    if provider == "Google Gemini" or (provider == "Groq Cloud" and embed_engine.startswith("Gemini")):
                        # Embed Query using Gemini
                        key_to_use = active_api_key if provider == "Google Gemini" else gemini_key_for_embed
                        genai.configure(api_key=key_to_use)
                        query_response = genai.embed_content(
                            model="models/text-embedding-004",
                            content=user_query
                        )
                        query_vector = np.array(query_response['embedding'][0])
                        
                        # Cosine Similarity Dot Product
                        q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
                        norms = np.linalg.norm(db_index, axis=1, keepdims=True)
                        norms[norms == 0] = 1.0
                        normalized_db = db_index / norms
                        similarities = np.dot(normalized_db, q_norm)
                    else:
                        # Local TF-IDF Cosine Similarity
                        similarities = query_tfidf(user_query, db_index)
                    
                    # Sort top indices
                    top_indices = np.argsort(similarities)[::-1][:top_k]
                    
                    # Create Context Prompt and Citations
                    context_chunks = []
                    sources = []
                    for idx in top_indices:
                        chunk = st.session_state.chunks[idx]
                        score = float(similarities[idx]) * 100 # Match percentage
                        
                        context_chunks.append(
                            f"--- START CHUNK (Source: {chunk['source']}, Page: {chunk['page_num']}) ---\n"
                            f"{chunk['text']}\n"
                            f"--- END CHUNK ---\n"
                        )
                        sources.append({
                            "source": chunk["source"],
                            "page_num": chunk["page_num"],
                            "text": chunk["text"][:220] + "...", # preview
                            "score": score
                        })
                    
                    context_str = "\n\n".join(context_chunks)
                    
                    # System instruction prompt
                    system_prompt = (
                        "You are an expert PDF Chat Assistant. You will be provided with context chunks from the user's uploaded PDF documents. "
                        "Your goal is to answer the user's question accurately, clearly, and concisely, relying ONLY on the provided context.\n\n"
                        "Follow these rules strictly:\n"
                        "1. Base your answer solely on the context provided. Do not use outside knowledge unless it is minor clarification.\n"
                        "2. If the answer cannot be found in the context, state clearly: 'I cannot find the answer to this question in the uploaded documents.' Do not make up information.\n"
                        "3. Provide inline citations in your response referencing the document and page number. Format as [Doc: filename, Page: X].\n"
                        "4. Keep your formatting clean, readable, and structured using markdown.\n"
                    )
                    
                    prompt_text = f"Context from documents:\n{context_str}\n\nQuestion: {user_query}"
                    
                    # 2. Text generation
                    response_text = generate_response(
                        provider=provider,
                        model_name=selected_model,
                        api_key=active_api_key,
                        system_prompt=system_prompt,
                        prompt_text=prompt_text,
                        temperature=temperature
                    )
                    
                    # Render Response
                    st.markdown(response_text)
                    
                    # Render expandable citations
                    with st.expander("📚 View Referenced Sources"):
                        for idx, src in enumerate(sources):
                            st.markdown(f"""
                            <div class="citation-container">
                                <div class="citation-header">Source {idx+1}: {src['source']} (Page {src['page_num']}) • Match Score: {src['score']:.1f}%</div>
                                <div class="citation-snippet">"{src['text']}"</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    # Append response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    st.markdown("Sorry, I encountered an error while processing your request. Please check your API key / model selections.")

    # Show Download Chat Button
    if st.session_state.chat_history:
        chat_txt = ""
        for msg in st.session_state.chat_history:
            chat_txt += f"{msg['role'].upper()}: {msg['content']}\n\n"
        
        st.download_button(
            label="💾 Download Conversation Transcript",
            data=chat_txt,
            file_name="pdf_chat_transcript.txt",
            mime="text/plain"
        )
