# from fastapi import FastAPI
# from api.endpoints import recommend, history

# app = FastAPI(
#     title="Anime Recommendation System API",
#     description="API for providing anime recommendations using LangChain and Gemini",
#     version="1.0.0"
# )

# app.include_router(recommend.router, prefix="/recommend", tags=["Recommendation"])
# app.include_router(history.router, prefix="/history", tags=["User History"])

# @app.get("/")
# def root():
#     return {"message": "Anime Recommendation System API"}

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import RecommendationRequest, HistoryRecommendationRequest, AnimeRecommendation
from recommendation import query_based_recommendation, history_based_recommendation

app = FastAPI(
    title="Anime Recommendation System API",
    description="API for providing anime recommendations using LangChain and Gemini",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Anime Recommendation System API"}

@app.post("/recommendation", response_model=list[AnimeRecommendation])
async def get_recommendation(request: RecommendationRequest):
    try:
        recommendations = query_based_recommendation(
            query=request.query,
            n_results=request.n_results,
            personalized=request.personalized,
            user_id=request.user_id
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/history-recommendation", response_model=list[AnimeRecommendation])
async def get_history_recommendation(request: HistoryRecommendationRequest):
    try:
        recommendations = history_based_recommendation(
            user_id=request.user_id,
            n_results=request.n_results
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))