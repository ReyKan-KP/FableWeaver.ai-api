from pydantic import BaseModel, Field
from typing import List, Optional

class QueryFilter(BaseModel):
    genres: Optional[List[str]] = Field(None, description="List of genres to filter by")
    year_start: Optional[int] = Field(None, description="Start year for filtering")
    year_end: Optional[int] = Field(None, description="End year for filtering")
    seasons: Optional[List[str]] = Field(None, description="List of seasons (spring, summer, fall, winter)")
    rating_min: Optional[float] = Field(None, description="Minimum rating")
    rating_max: Optional[float] = Field(None, description="Maximum rating")
    description_keywords: Optional[List[str]] = Field(None, description="Keywords to look for in description")

class RecommendationRequest(BaseModel):
    query: str
    n_results: int = Field(default=5, ge=1, le=20)
    personalized: bool = False
    user_id: Optional[str] = None

class HistoryRecommendationRequest(BaseModel):
    user_id: str
    n_results: int = Field(default=5, ge=1, le=20)

class AnimeScore(BaseModel):
    cosine_similarity: float
    feedback_score: float
    normalized_score: float
    combined_score: float

class AnimeRecommendation(BaseModel):
    title: str
    description: str
    rating: float
    year: str
    season: str
    genres: List[str]
    image_url: Optional[str] = None
    scores: AnimeScore

