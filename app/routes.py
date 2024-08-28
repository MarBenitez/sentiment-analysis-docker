from flask import Blueprint, render_template, request, jsonify
from app.model import predict_sentiment, analyze_user_tweets, analyze_hashtag_tweets

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    text = None
    if request.method == 'POST':
        text = request.form.get('text')
        if text:
            sentiment = predict_sentiment(text)
        else:
            sentiment = "No text provided"
    return render_template('index.html', sentiment=sentiment, text=text)

@main.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = predict_sentiment(text)
    return jsonify({"label": result}), 200

@main.route('/analyze_user_tweets', methods=['POST'])
def analyze_user():
    data = request.get_json(force=True)
    username = data.get("username", "")
    tweet_count = int(data.get("count", 10))
    if not username:
        return jsonify({"error": "No username provided"}), 400
    
    tweets = analyze_user_tweets(username, tweet_count)
    if tweets is not None:
        return jsonify({"tweets": tweets}), 200
    else:
        return jsonify({"error": "Failed to analyze user tweets"}), 500

@main.route('/analyze_hashtag_tweets', methods=['POST'])
def analyze_hashtag():
    data = request.get_json(force=True)
    hashtag = data.get("hashtag", "")
    tweet_count = int(data.get("count", 10))
    if not hashtag:
        return jsonify({"error": "No hashtag provided"}), 400
    
    tweets = analyze_hashtag_tweets(hashtag, tweet_count)
    if tweets is not None:
        return jsonify({"tweets": tweets}), 200
    else:
        return jsonify({"error": "Failed to analyze hashtag tweets"}), 500
