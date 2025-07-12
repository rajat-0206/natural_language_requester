"""
Index utilities for building and managing API search indices.
"""

import json
import torch
import faiss
import numpy as np
import os
import re
try:
    import scann
    SCANN_AVAILABLE = True
except ImportError:
    print("Warning: SCANN not available. Using scikit-learn NearestNeighbors as fallback.")
    from sklearn.neighbors import NearestNeighbors
    SCANN_AVAILABLE = False
from transformers import AutoTokenizer, AutoModel

# Cache directory
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache file paths
INDEX_FILE = os.path.join(CACHE_DIR, "api_index.bin")
METADATA_FILE = os.path.join(CACHE_DIR, "api_metadata.json")
VERSION_FILE = os.path.join(CACHE_DIR, "schema_version.txt")
TFIDF_FILE = os.path.join(CACHE_DIR, "tfidf_matrix.npy")
TFIDF_VECTORIZER_FILE = os.path.join(CACHE_DIR, "tfidf_vectorizer.pkl")

# SCANN cache file paths
SCANN_INDEX_FILE = os.path.join(CACHE_DIR, "scann_api_index.pkl")

def get_schema_version():
    """Get the current schema version."""
    try:
        with open("schema.json", "r") as f:
            schema = json.load(f)
        return schema.get('version', '1.0')
    except FileNotFoundError:
        return '1.0'

def embed(texts):
    """Generate embeddings for the given texts using the BERT model."""
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    
    # Tokenize and encode
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    # Generate embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Use mean pooling
    attention_mask = encoded_input['attention_mask']
    embeddings = model_output.last_hidden_state * attention_mask.unsqueeze(-1)
    embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    
    return embeddings.numpy()

def build_index(schema_entries):
    """Build FAISS index for API search."""
    summaries = []
    metadata = []
    for entry in schema_entries["apis"]:
        # Start with operationId as it's the most descriptive
        operation_id = entry.get('operationId', '')
        description = entry.get("description", "")
        method = entry.get('method', '')
        
        # Create a weighted summary that emphasizes the operation's purpose
        summary_parts = []
        
        # Add HTTP method with HIGHEST weight and descriptive action words
        method_action_map = {
            'GET': 'fetch retrieve get read obtain',
            'POST': 'create add new insert',
            'PATCH': 'update modify edit change',
            'PUT': 'update replace modify edit change',
            'DELETE': 'remove delete destroy eliminate'
        }
        
        # Add method with very high weight (repeated many times for emphasis)
        if method in method_action_map:
            # Repeat method 8 times for strongest emphasis
            summary_parts.extend([method] * 8)
            # Add descriptive action words for the method
            summary_parts.extend(method_action_map[method].split() * 4)
        
        # Add operationId with high weight (repeated for emphasis)
        if operation_id:
            # Repeat operationId 5 times for emphasis
            summary_parts.extend([operation_id] * 5)
            
            # Add operationId with spaces between words for better matching
            # e.g., "createEvent" becomes "create event"
            spaced_operation = ' '.join(re.findall('[A-Z][a-z]*', operation_id)).lower()
            summary_parts.extend([spaced_operation] * 3)
        
        # Add description if available
        if description:
            summary_parts.append(description)
            
        # Add method and path with medium weight
        summary_parts.append(f"{method} {entry['path']}")
        
        # Add request body or query params with lowest weight
        if method == "GET":
            query_params = entry.get("parameters", [])
            if query_params:
                summary_parts.append(f"Query params: {query_params}")
        else:
            request_body = entry.get("requestBody", {})
            if request_body:
                summary_parts.append(f"Request body: {request_body}")
        
        # Join all parts with spaces
        summary = " ".join(summary_parts)
        summaries.append(summary)
        metadata.append(entry)

    # Build embedding index
    embeddings = embed(summaries)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, metadata, summaries, None, None

def load_cached_index():
    """Load the cached index and metadata if they exist and schema version hasn't changed."""
    if not all(os.path.exists(f) for f in [INDEX_FILE, METADATA_FILE, VERSION_FILE]):
        return None, None, None, None, None
    
    # Check if schema version has changed
    with open(VERSION_FILE, 'r') as f:
        cached_version = f.read().strip()
    current_version = get_schema_version()
    if cached_version != current_version:
        print(f"Schema version changed from {cached_version} to {current_version}")
        return None, None, None, None, None
    
    # Load cached data
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        return index, metadata, None, None, None
    except Exception as e:
        print(f"Error loading cached index: {e}")
        return None, None, None, None, None

def save_index_to_cache(index, metadata, tfidf_matrix, vectorizer):
    """Save the index and metadata to cache."""
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f)
        with open(VERSION_FILE, 'w') as f:
            f.write(get_schema_version())
        if tfidf_matrix is not None:
            np.save(TFIDF_FILE, tfidf_matrix)
        if vectorizer is not None:
            import pickle
            with open(TFIDF_VECTORIZER_FILE, 'wb') as f:
                pickle.dump(vectorizer, f)
    except Exception as e:
        print(f"Error saving index to cache: {e}")

def build_scann_index(schema_entries):
    """Build SCANN index for API search."""
    if not SCANN_AVAILABLE:
        raise ImportError("SCANN is not available. Please install it first.")
    
    summaries = []
    metadata = []
    for entry in schema_entries["apis"]:
        # Start with operationId as it's the most descriptive
        operation_id = entry.get('operationId', '')
        description = entry.get("description", "")
        method = entry.get('method', '')
        
        # Create a weighted summary that emphasizes the operation's purpose
        summary_parts = []
        
        # Add HTTP method with HIGHEST weight and descriptive action words
        method_action_map = {
            'GET': 'fetch retrieve get read obtain',
            'POST': 'create add new insert',
            'PATCH': 'update modify edit change',
            'PUT': 'update replace modify edit change',
            'DELETE': 'remove delete destroy eliminate'
        }
        
        # Add method with very high weight (repeated many times for emphasis)
        if method in method_action_map:
            # Repeat method 8 times for strongest emphasis
            summary_parts.extend([method] * 8)
            # Add descriptive action words for the method
            summary_parts.extend(method_action_map[method].split() * 4)
        
        # Add operationId with high weight (repeated for emphasis)
        if operation_id:
            # Repeat operationId 5 times for emphasis
            summary_parts.extend([operation_id] * 5)
            
            # Add operationId with spaces between words for better matching
            # e.g., "createEvent" becomes "create event"
            spaced_operation = ' '.join(re.findall('[A-Z][a-z]*', operation_id)).lower()
            summary_parts.extend([spaced_operation] * 3)
        
        # Add description if available
        if description:
            summary_parts.append(description)
            
        # Add method and path with medium weight
        summary_parts.append(f"{method} {entry['path']}")
        
        # Add request body or query params with lowest weight
        if method == "GET":
            query_params = entry.get("parameters", [])
            if query_params:
                summary_parts.append(f"Query params: {query_params}")
        else:
            request_body = entry.get("requestBody", {})
            if request_body:
                summary_parts.append(f"Request body: {request_body}")
        
        # Join all parts with spaces
        summary = " ".join(summary_parts)
        summaries.append(summary)
        metadata.append(entry)

    # Build embedding index
    embeddings = embed(summaries)
    
    # Build SCANN index
    scann_index = scann.scann_ops_pybind.builder(embeddings, 10, "dot_product").tree(
        num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000
    ).score_ah(2, anisotropic_quantization_threshold=0.2).build()
    
    return scann_index, metadata, summaries, None, None

def load_cached_scann_index():
    """Load the cached SCANN index and metadata if they exist and schema version hasn't changed."""
    if not os.path.exists(SCANN_INDEX_FILE) or not all(os.path.exists(f) for f in [METADATA_FILE, VERSION_FILE]):
        return None, None, None, None, None
    
    # Check if schema version has changed
    with open(VERSION_FILE, 'r') as f:
        cached_version = f.read().strip()
    current_version = get_schema_version()
    if cached_version != current_version:
        print(f"Schema version changed from {cached_version} to {current_version}")
        return None, None, None, None, None
    
    # Load cached data
    try:
        import pickle
        with open(SCANN_INDEX_FILE, 'rb') as f:
            scann_index = pickle.load(f)
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        return scann_index, metadata, None, None, None
    except Exception as e:
        print(f"Error loading cached SCANN index: {e}")
        return None, None, None, None, None

def save_scann_index_to_cache(scann_index, metadata, tfidf_matrix, vectorizer):
    """Save the SCANN index and metadata to cache."""
    try:
        import pickle
        with open(SCANN_INDEX_FILE, 'wb') as f:
            pickle.dump(scann_index, f)
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f)
        with open(VERSION_FILE, 'w') as f:
            f.write(get_schema_version())
    except Exception as e:
        print(f"Error saving SCANN index to cache: {e}")

def search_apis_scann(query, scann_index, metadata, tfidf_matrix, vectorizer, top_k=3):
    """Search APIs using SCANN index."""
    if not SCANN_AVAILABLE:
        raise ImportError("SCANN is not available. Please install it first.")
    
    # Generate query embedding
    query_embedding = embed([query])
    
    # Search using SCANN
    neighbors, distances = scann_index.search(query_embedding, top_k)
    
    # Format results
    results = []
    for i, (neighbor_idx, distance) in enumerate(zip(neighbors[0], distances[0])):
        if neighbor_idx < len(metadata):
            results.append({
                'api': metadata[neighbor_idx],
                'score': float(distance),
                'rank': i + 1
            })
    
    return results 