import json 
from pathlib import Path 
import re 

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/"data"
MOVIES_PATH = DATA_PATH/"movies.json"
STOPWORDS_PATH = DATA_PATH/"stopwords.txt"

CACHE_PATH = PROJECT_ROOT/"cache"

def load_movies():
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH, "r") as f:
        stopwords = [line.strip() for line in f]
    return stopwords