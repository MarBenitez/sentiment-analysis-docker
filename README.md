# Sentiment Analysis API

This project is a sentiment analysis web application powered by Flask and Hugging Face's Transformers library. It allows users to submit text and recieve an analysis indicating whether the sentiment is positive or negative. The project is containerized using Docker, making it easy to deploy and scale in production environments.

![App Screenshot](https://github.com/MarBenitez/sentiment-analysis-docker/blob/main/app/static/screeshot-app.png)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running in Development](#running-in-development)
  - [Running in Production](#running-in-production)
- [Usage](#usage)
- [Deployment](#deployment)
- [License](#license)
- [Contact](#contact)

## Overview

The Sentiment Analysis API provides a simple web interface where users can input text and receive an instant sentiment analysis. The backend is powered by a pre-trained DistilBERT model from Hugging Face, fine-tuned on sentiment analysis.

## Features

- **Sentiment Analysis:** Analyze the sentiment of a given text (positive or negative).
- **Responsive UI:** A clean and responsive user interface built with HTML and CSS.
- **Dockerized:** Easily deployable using Docker, with configurations for both development and production environments.
- **Scalable:** The app is set up to run with Gunicorn for handling multiple requests in a production setting.

## Technologies Used

- **Flask:** Python micro-framework for building web applications.
- **Transformers:** Hugging Face library for state-of-the-art NLP.
- **Docker:** Containerization platform to deploy the application.
- **Gunicorn:** WSGI HTTP server for Python web applications.
- **HTML/CSS:** Frontend structure and styling.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Docker
- Docker Compose
- Python 3.9+

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/MarBenitez/sentiment-analysis-docker
   cd sentiment-analysis-docker
   ```

2. **Build the Docker image:**

   For development:

   ```bash
   docker-compose build
   ```

   For production:

   ```bash
   docker-compose -f docker-compose.prod.yml build
   ```

### Running in Development

1. **Start the development server:**

   ```bash
   docker-compose up
   ```

2. **Access the application:**

   Navigate to `http://localhost:8080` in your web browser.

### Running in Production

1. **Start the production server:**

   ```bash
   docker-compose -f docker-compose.prod.yml up
   ```

2. **Access the application:**

   Navigate to `http://your-server-ip:8080` in your web browser.

## Usage

1. **Submit Text:**
   - Enter your text in the input field on the home page.
   - Click the "Analyze" button.

2. **View Results:**
   - The result will be displayed below the input field, showing whether the sentiment is positive or negative.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

Feel free to reach out if you have any questions or want to connect:

- **Email:** mar27benitez@gmail.com
- **LinkedIn:** [Mar Benitez](https://www.linkedin.com/in/mar-benitez-506910192/)
- **GitHub:** [Mar Benitez](https://github.com/MarBenitez/)
