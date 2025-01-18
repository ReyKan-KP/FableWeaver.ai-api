from supabase import create_client, Client
import os
from typing import List, Dict, Any
from ast import literal_eval
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local", override=True)
SUPABASE_URL = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
SUPABASE_KEY = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_anime_details(anime_ids: List[str]) -> List[Dict[str, Any]]:
    try:
        response = supabase.table('anime').select('*').in_('id', anime_ids).execute()
        return response.data
    except Exception as e:
        print(f"Error fetching anime details: {e}")
        return []

def get_user_history(user_id: str) -> List[str]:
    try:
        response = supabase.table('user').select('user_watched_list').eq('user_id', user_id).execute()
        if response.data:
            user = response.data[0]
            watched_list = user.get('user_watched_list', '')
            return literal_eval(watched_list)
        return []
    except Exception as e:
        print(f"Error fetching user history: {e}")
        return []

def get_anime_feedback(anime_id: str) -> float:
    try:
        response = supabase.table('anime').select('num_favorites', 'num_list_users', 'feedback').eq('id', anime_id).execute()
        if response.data:
            anime = response.data[0]
            num_favorites = float(anime.get('num_favorites', 0))
            num_lists = float(anime.get('num_list_users', 1))
            if num_lists == 0:
                num_lists = 1
            user_feedback = anime.get('feedback', 1)
            return num_favorites / num_lists + user_feedback
        return 0.0
    except Exception as e:
        print(f"Error getting anime feedback: {e}")
        return 0.0

def get_anime_normalized_rank(anime_id: str) -> float:
    try:
        response = supabase.table('anime').select('rank').eq('id', anime_id).execute()
        if response.data:
            anime = response.data[0]
            rank = float(anime.get('rank', 0))
            normalized_rank = (get_total_docs() - rank) / get_total_docs()
            return normalized_rank
        return 0.0
    except Exception as e:
        print(f"Error getting anime rank: {e}")
        return 0.0

def get_anime_image_url(anime_id: str) -> str:
    try:
        response = supabase.table('anime').select('image_url').eq('id', anime_id).execute()
        if response.data:
            anime = response.data[0]
            return anime.get('image_url', '')
        return ''
    except Exception as e:
        print(f"Error getting anime image: {e}")
        return ''

def get_total_docs() -> int:
    return 24012  # Fallback to a fixed number to avoid division by zero

