# Sentiment Analysis Web App

A simple web application built with Flask that performs sentiment analysis on user-input text using a pre-trained scikit-learn model.

## Features

* **Real-time Prediction:** Analyze text sentiment instantly via a web interface.
* **Machine Learning Model:** Utilizes a scikit-learn pipeline (Vectorization + Naive Bayes) for classification.
* **Simple Web Interface:** A straightforward front-end to input text and view the results.

## Local Setup

To run this project locally, follow these steps.

### Prerequisites

* Python 3.10 or newer
* `pip` (Python package installer)

### 1. Installation

1.  Clone the repository to your local machine (if you haven't already).
2.  Navigate to the project directory:
    ```bash
    cd sentiment-analysis-web-app
    ```
3.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Running the Application

Start the Flask development server:

```bash
python app.py
