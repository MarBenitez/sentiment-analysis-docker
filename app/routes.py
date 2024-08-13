
# define the application paths and logic for handling requests

from flask import Blueprint, render_template, request, jsonify
from app.model import predict_sentiment

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
