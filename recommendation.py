from typing import List, Dict, Any
from models import AnimeRecommendation, AnimeScore, QueryFilter
from database import get_anime_details, get_user_history, get_anime_feedback, get_anime_normalized_rank, get_anime_image_url, get_total_docs
from utils import parse_query, filter_metadata, calculate_normalized_evaluation, combine_scores, extract_genres_from_string
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local", override=True)
# Initialize Pinecone
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index("embeddings-animes")

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def query_based_recommendation(query: str, n_results: int = 5, personalized=False, user_id=None) -> List[AnimeRecommendation]:
    try:
        if personalized and user_id is not None:
            watch_history = get_user_history(user_id)
            if watch_history:
                history_details = get_anime_details(watch_history)
                query_parts = [
                    f"{anime.get('title', '')} {' '.join(extract_genres_from_string(anime.get('genres', '[]')))}"
                    for anime in history_details if anime.get('title', '').strip()
                ]
                query = f"{query} {' '.join(query_parts)}"

        filters = parse_query(query)
        query_embedding = embeddings.embed_query(query)

        results = index.query(
            vector=query_embedding,
            top_k=n_results * 10,
            include_metadata=True
        )

        total_docs = get_total_docs()
        ranked_results = []

        for match in results.matches:
            if filter_metadata(match.metadata, filters):
                anime_id = match.metadata['id']
                feedback = get_anime_feedback(anime_id)
                normalized_rank = get_anime_normalized_rank(anime_id)
                image_url = get_anime_image_url(anime_id)
                rating = float(match.metadata.get('rating', 1))

                normalized_score = calculate_normalized_evaluation(
                    rating=rating,
                    feedback=feedback,
                    normalized_rank=normalized_rank,
                    total_docs=total_docs
                )

                combined_score = combine_scores(
                    cosine_similarity=match.score,
                    normalized_score=normalized_score
                )

                ranked_results.append(AnimeRecommendation(
                    title=match.metadata.get('title', ''),
                    description=match.metadata.get('description', ''),
                    rating=rating,
                    year=str(match.metadata.get('year', '')),
                    season=match.metadata.get('season', ''),
                    genres=extract_genres_from_string(match.metadata.get('genres', '[]')),
                    image_url=image_url,
                    scores=AnimeScore(
                        cosine_similarity=round(match.score, 4),
                        feedback_score=round(feedback, 4),
                        normalized_score=round(normalized_score, 4),
                        combined_score=round(combined_score, 4)
                    )
                ))

        ranked_results.sort(key=lambda x: x.scores.combined_score, reverse=True)
        return ranked_results[:n_results]

    except Exception as e:
        print(f"Error in query_based_recommendation: {e}")
        return []

def history_based_recommendation(user_id: str, n_results: int = 5) -> List[AnimeRecommendation]:
    try:
        watch_history = get_user_history(user_id)
        if not watch_history:
            print(f"No watch history found for user {user_id}")
            return []

        history_details = get_anime_details(watch_history)
        if not history_details:
            print(f"Could not fetch anime details for watch history")
            return []

        query_parts = [
            f"{anime.get('title', '')} {' '.join(extract_genres_from_string(anime.get('genres', '[]')))}"
            for anime in history_details if anime.get('title', '').strip()
        ]

        if not query_parts:
            print("No valid anime data found in history")
            return []

        combined_query = " ".join(query_parts)
        recommendations = query_based_recommendation(combined_query, n_results)

        return [rec for rec in recommendations if rec.title not in set(anime.get('title', '') for anime in history_details)]

    except Exception as e:
        print(f"Error in history_based_recommendation: {e}")
        return []

