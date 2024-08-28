import os
from flask import Flask
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

def create_app():
    app = Flask(__name__)
    from app.routes import main
    app.register_blueprint(main)
    return app
