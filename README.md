# Multi-Provider PDF Chatbot
<img width="1906" height="917" alt="image" src="https://github.com/user-attachments/assets/7fbd5721-e671-40a6-85ed-bf2ea997a1a5" />

A premium, highly interactive Streamlit web application that lets you upload multiple PDF documents and chat with them using either **Google Gemini** or **Groq Cloud** models. 

## Features
- **Multi-Provider Support**: Choose between **Google Gemini** (`gemini-2.5-flash`, `gemini-2.5-pro`) and **Groq Cloud** (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, `gemma2-9b-it`).
- **Flexible Document Searching**:
  - **Gemini Embeddings**: Performs deep semantic semantic search using `text-embedding-004` (requires Gemini API Key).
  - **TF-IDF Keyword Search**: A pure Python/NumPy implementation of TF-IDF Cosine Similarity that is completely local, free, and requires no API key. Excellent for offline or low-cost testing.
- **Exact Citation Mapping**: Responses include source page citations and an expandable section to view matching source text snippets.
- **Multi-PDF Upload**: Upload multiple PDF files at once.
- **Visual Process Flow**: Real-time status update checklist for text extraction, chunking, and embedding/vector construction.
- **Dashboard Stats**: Real-time summary counts of files, total pages, chunks, and word counts.
- **Transcript Export**: Download your chat history as a formatted text file.
- **Modern UI**: Styled with responsive CSS, glassmorphism sidebar, custom cards, and custom scroll bars.

---

## Installation & Setup

1. **Verify Python is Installed**:
   Make sure you have Python 3.9+ installed.

2. **Install Dependencies**:
   Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Get API Keys**:
   - **Groq API Key**: Obtain a key from the [Groq Console](https://console.groq.com/).
   - **Gemini API Key**: Obtain a key from [Google AI Studio](https://aistudio.google.com/).

4. **Environment Configuration (Optional)**:
   Create a `.env` file in the root directory and paste your keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   *Note: If no `.env` file is present, you can enter the API keys directly inside the app sidebar.*

---

## Running the Application

To launch the chatbot, run the following command in your terminal:

```bash
streamlit run app.py
```

This will spin up a local development server and automatically open a new tab in your browser (usually at `http://localhost:8501`).
