# load the pre-trained model and define the function that predicts the sentiment of a given text

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def load_pipeline():
    # Option 1: Use pipeline directly
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
    # Option 2: Load the model and the tokenizer directly
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"Error loading model directly: {e}")
        sentiment_pipeline = None
    
    return sentiment_pipeline

# Try to load pipeline first
sentiment_pipeline = load_pipeline()

# If pipeline could not be loaded, try to load model directly
if not sentiment_pipeline:
    sentiment_pipeline = load_model_directly()

def predict_sentiment(text):
    if sentiment_pipeline:
        result = sentiment_pipeline(text)
        return result[0]['label']
    else:
        return "Model could not be loaded."

