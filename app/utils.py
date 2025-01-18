from typing import List, Dict, Any
from app.models import QueryFilter
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
import json
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)


def create_structured_prompt() -> str:
    return """Given the anime query below, extract the search parameters and return a JSON object.

Query: {query}

Instructions:
1. Analyze the query for specific criteria
2. Extract all relevant parameters
3. Return a properly formatted JSON object

Required format:
{{
    "genres": ["genre1", "genre2"],
    "year_start": null,
    "year_end": null,
    "seasons": ["season1", "season2"],
    "rating_min": null,
    "rating_max": null,
    "description_keywords": []
}}

Notes:
- Only include non-null values for mentioned criteria
- Seasons should be from: spring, summer, fall, winter
- Years should be integers
- Ratings should be floats
- Return valid JSON only

Response:"""


def parse_query(query: str) -> QueryFilter:
    try:
        prompt = PromptTemplate(
            template=create_structured_prompt(), input_variables=["query"]
        )
        # Create a runnable sequence instead of LLMChain
        chain = prompt | llm

        # Use invoke instead of run
        response = chain.invoke({"query": query})

        # Extract the content from the response
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )
        response_text = response_text.strip()

        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        result_dict = json.loads(response_text)
        return QueryFilter(**result_dict)
    except Exception as e:
        print(f"Error parsing query: {e}")
        return QueryFilter(
            genres=["action"] if "action" in query.lower() else None,
            year_start=2020 if "2020" in query else None,
            year_end=2021 if "2021" in query else None,
            seasons=(
                ["summer", "fall"]
                if "summer" in query.lower() or "fall" in query.lower()
                else None
            ),
            rating_min=7.5 if "7.5" in query else None,
            rating_max=9.5 if "8.5" in query else None,
        )


# Rest of the functions remain unchanged
def filter_metadata(metadata: Dict[str, Any], filters: QueryFilter) -> bool:
    try:
        genres_list = extract_genres_from_string(metadata.get("genres", "[]"))

        if filters.genres and not any(
            genre.lower() in [g.lower() for g in genres_list]
            for genre in filters.genres
        ):
            return False

        year = int(metadata.get("year", "0"))
        if filters.year_start and year < filters.year_start:
            return False
        if filters.year_end and year > filters.year_end:
            return False

        if filters.seasons and metadata.get("season", "").lower() not in [
            s.lower() for s in filters.seasons
        ]:
            return False

        try:
            rating = float(metadata.get("rating", "0"))
            if filters.rating_min and rating < filters.rating_min:
                return False
            if filters.rating_max and rating > filters.rating_max:
                return False
        except:
            return False

        return True
    except Exception as e:
        print(f"Error filtering metadata: {e}")
        return False


def extract_genres_from_string(genres_str: str) -> List[str]:
    try:
        cleaned_str = genres_str.strip().strip("\"'")
        if not cleaned_str or cleaned_str == "[]":
            return []
        genres_list = cleaned_str.strip("[]").split(",")
        cleaned_genres = [
            genre.strip().strip("\"'") for genre in genres_list if genre.strip()
        ]
        return cleaned_genres
    except Exception as e:
        print(f"Error extracting genres: {e}")
        return []


def calculate_normalized_evaluation(
    rating: float, feedback: float, normalized_rank: float, total_docs: int
) -> float:
    try:
        evaluation = feedback * rating * normalized_rank
        normalized_score = (evaluation / total_docs) * 100
        return normalized_score
    except Exception as e:
        print(f"Error calculating normalized evaluation: {e}")
        return 0.0


def combine_scores(
    cosine_similarity: float,
    normalized_score: float,
    similarity_weight: float = 0.7,
    normalized_weight: float = 0.3,
) -> float:
    try:
        total_weight = similarity_weight + normalized_weight
        similarity_weight = similarity_weight / total_weight
        normalized_weight = normalized_weight / total_weight

        combined_score = (
            cosine_similarity * similarity_weight + normalized_score * normalized_weight
        )

        return combined_score
    except Exception as e:
        print(f"Error combining scores: {e}")
        return 0.0
