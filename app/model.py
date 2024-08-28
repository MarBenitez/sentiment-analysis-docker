import os
import tweepy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Acceder a las credenciales de la API de Twitter desde las variables de entorno
consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_SECRET")

# Autenticación en la API de Twitter
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Probar la conexión a la API de Twitter
try:
    public_tweets = api.user_timeline(count=1)  # Cambia home_timeline a user_timeline para evitar el uso de un endpoint restringido
    for tweet in public_tweets:
        print(tweet.text)
    print("Conexión a la API de Twitter exitosa.")
except tweepy.TweepyException as e:  # Cambia TweepError por TweepyException
    print(f"Error al acceder a la API de Twitter: {e}")

# Funciones para cargar el modelo de análisis de sentimientos
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

# Función para obtener y analizar los tweets de un usuario específico
def analyze_user_tweets(username, count=10):
    try:
        tweets = api.user_timeline(screen_name=username, count=count)
        analyzed_tweets = []
        for tweet in tweets:
            sentiment = predict_sentiment(tweet.text)
            analyzed_tweets.append({'text': tweet.text, 'sentiment': sentiment})
        return analyzed_tweets
    except tweepy.TweepyException as e:  # Cambia TweepError por TweepyException
        error_message = f"Error al obtener los tweets del usuario: {e}"
        print(error_message)
        return {"error": error_message}

# Función para obtener y analizar tweets relacionados con un hashtag específico
def analyze_hashtag_tweets(hashtag, count=10):
    try:
        tweets = tweepy.Cursor(api.search_tweets, q=hashtag, lang="en").items(count)
        analyzed_tweets = []
        for tweet in tweets:
            sentiment = predict_sentiment(tweet.text)
            analyzed_tweets.append({'text': tweet.text, 'sentiment': sentiment})
        return analyzed_tweets
    except tweepy.TweepyException as e:  # Cambia TweepError por TweepyException
        error_message = f"Error al obtener los tweets del hashtag: {e}"
        print(error_message)
        return {"error": error_message}
