from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# ✅ Load the entire pipeline (vectorizer + model)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.form['text']
    prediction = model.predict([user_text])[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    return render_template('index.html', prediction=sentiment, user_input=user_text)

if __name__ == '__main__':
    app.run(debug=True)
