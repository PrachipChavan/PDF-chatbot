import numpy as np
import math
import re

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def index_tfidf(chunks):
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

def test_tfidf_and_embeddings():
    print("Testing TF-IDF Core Indexing and Querying...")
    
    dummy_chunks = [
        {"text": "The quick brown fox jumps over the lazy dog", "page_num": 1, "source": "doc1.pdf"},
        {"text": "Artificial Intelligence and Machine Learning are transforming technology", "page_num": 2, "source": "doc1.pdf"},
        {"text": "Streamlit is an amazing library to build web applications with Python", "page_num": 3, "source": "doc1.pdf"}
    ]
    
    # 1. Test indexing
    index = index_tfidf(dummy_chunks)
    assert len(index["vocab_idx"]) > 0, "Vocab index empty"
    assert index["vectors"].shape == (3, len(index["vocab_idx"])), "Vector matrix dimensions mismatch"
    print("[SUCCESS] TF-IDF Indexing passed!")
    
    # 2. Test query similarity
    scores = query_tfidf("Python applications", index)
    print("-> TF-IDF similarity scores for query 'Python applications':", scores)
    # The 3rd document contains "Python" and "applications" (or parts of them), so it should rank first.
    assert scores[2] > scores[0], "Ranking failed: 3rd doc should have higher score"
    assert scores[2] > scores[1], "Ranking failed: 3rd doc should have higher score"
    print("[SUCCESS] TF-IDF Query Cosine Similarity ranking passed!")
    
    print("\nAll unit tests passed successfully!")

if __name__ == "__main__":
    test_tfidf_and_embeddings()
