import os
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Acceder a las credenciales de la API de Twitter desde las variables de entorno
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

def verify_twitter_api_connection():
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    response = requests.get("https://api.twitter.com/2/tweets", headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error connecting to Twitter API: {response.status_code} - {response.text}")
    else:
        print("Twitter API connection verified successfully.")

# Llamar a la función de verificación al inicio del script
try:
    verify_twitter_api_connection()
except Exception as e:
    print(f"Warning: {e}")

# Configurar el modelo de análisis de sentimientos
def load_pipeline():
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        )
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        sentiment_pipeline = None
    
    return sentiment_pipeline

def load_model_directly():
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"Error loading model directly: {e}")
        sentiment_pipeline = None
    
    return sentiment_pipeline

sentiment_pipeline = load_pipeline()

if not sentiment_pipeline:
    sentiment_pipeline = load_model_directly()

def predict_sentiment(text):
    if sentiment_pipeline:
        result = sentiment_pipeline(text)
        return result[0]['label']
    else:
        return "Model could not be loaded."

# Función para buscar tweets recientes usando la API v2 de Twitter
def search_recent_tweets(query, max_results=10):
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    query_params = {
        'query': query,
        'max_results': max_results
    }
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    response = requests.get(search_url, headers=headers, params=query_params)
    if response.status_code != 200:
        raise Exception(f"Error en la solicitud: {response.status_code} - {response.text}")
    return response.json()

def analyze_user_tweets(username, count=10):
    query = f"from:{username}"
    tweets = search_recent_tweets(query, max_results=count)
    sentiments = [predict_sentiment(tweet["text"]) for tweet in tweets["data"]]
    return sentiments

def analyze_hashtag_tweets(hashtag, count=10):
    query = f"#{hashtag}"
    tweets = search_recent_tweets(query, max_results=count)
    sentiments = [predict_sentiment(tweet["text"]) for tweet in tweets["data"]]
    return sentiments
