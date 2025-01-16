"""
Anime Database Embeddings Generator

This script fetches anime data from Supabase, processes it, and generates 
embeddings using sentence-transformers, then stores them in Pinecone 
for vector search capabilities.
"""

import os
import pandas as pd
from supabase import create_client
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone as pc
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv

# Environment variables
load_dotenv(dotenv_path=".env.local", override=True)

SUPABASE_URL = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
SUPABASE_KEY = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

def fetch_anime_data(supabase_client, chunk_size: int = 1000) -> List[Dict]:
    """
    Fetch anime data from Supabase in chunks.
    
    Args:
        supabase_client: Initialized Supabase client
        chunk_size: Number of records to fetch per request
        
    Returns:
        List of anime records
    """
    all_animes = []
    total_records = 24012  # Total number of records to fetch
    
    for start in tqdm(range(0, total_records, chunk_size), desc="Fetching data"):
        response = supabase_client.table('anime').select('*').range(
            start, 
            start + chunk_size - 1
        ).execute()
        all_animes.extend(response.data)
        
    return all_animes

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the data for embedding generation.
    
    Args:
        df: Raw anime DataFrame
        
    Returns:
        Processed DataFrame ready for embedding generation
    """
    # Convert ID to string
    df['id'] = df['id'].astype(str)
    
    # Convert genres list to comma-separated string
    df['genres'] = df['genres'].apply(
        lambda x: ','.join(x) if isinstance(x, list) else x
    )
    
    # Combine text fields for embedding
    df['combined_text'] = (
        df['title'] + ' ' + 
        df['description'].fillna('') + ' ' + 
        df['genres'].fillna('')
    )
    df['combined_text'].fillna('', inplace=True)
    
    return df

def upsert_to_pinecone(batch: pd.DataFrame, 
                       index: pc.Index, 
                       embeddings: HuggingFaceEmbeddings) -> None:
    """
    Upload a batch of embeddings to Pinecone.
    
    Args:
        batch: Batch of records to process
        index: Pinecone index instance
        embeddings: Embedding model instance
    """
    try:
        # Generate embeddings for the batch
        embeds = embeddings.embed_documents(batch['combined_text'].tolist())
        ids = batch['id'].tolist()
        meta = batch[
            ['id', 'title', 'description', 'genres', 'year', 'season', 'rating']
        ].to_dict('records')
        
        # Upsert to Pinecone
        index.upsert(vectors=list(zip(ids, embeds, meta)), batch_size=100)
    except Exception as e:
        print(f"Error upserting batch: {e}")

def main():
    """Main execution function."""
    try:
        # Initialize Supabase client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase")
        
        # Fetch data
        print("Fetching anime data...")
        all_animes = fetch_anime_data(supabase)
        df = pd.DataFrame(all_animes)
        print(f"Total records fetched: {len(df)}")
        
        # Preprocess data
        print("Preprocessing data...")
        df = preprocess_data(df)
        
        # Initialize embeddings
        print("Initializing embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Initialize Pinecone
        # Uncomment these lines when setting up a new index
        # pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "embeddings-animes"
        # if index_name not in pc.list_indexes().names():
        #   pc.create_index(index_name, dimension=384, metric='cosine',   
        #           spec=ServerlessSpec(cloud='aws', region='us-east-1'))
        
        index = pc.Index(index_name)
        print("Connected to Pinecone")
        
        # Process and upload in batches
        batch_size = 100
        total_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
        
        print("Starting batch processing...")
        for i in tqdm(range(0, len(df), batch_size), total=total_batches):
            batch = df.iloc[i:i+batch_size]
            upsert_to_pinecone(batch, index, embeddings)
            
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
