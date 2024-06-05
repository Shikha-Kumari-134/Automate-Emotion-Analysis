import numpy as np
from flask import Flask, request, render_template
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create flask app
app = Flask(__name__)

# Load the pkl file
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def Home():
    return render_template("index.html", prediction_text="")

# Assuming 'model' is your trained machine learning model for emotion detection

# Create a tokenizer for text processing
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(["empty", "sadness", "enthusiasm", "neutral", "worry", "surprise",
                        "love sentiment", "fun", "hate", "happiness", "boredom", "relief", "anger"])

@app.route('/predict', methods=["POST"])
def predict():
    # Get input text from the POST request
    input_text = request.form['text']  # Assuming 'text' is the name of the text input field in your HTML form

    # Tokenize the input text using the tokenizer
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

    # Perform prediction using the model
    prediction = model.predict(padded_sequences)

    # Assuming 'prediction' is a numerical output representing emotions, convert it to a readable format
    emotion_mapping = {0: "Empty", 1: "Sadness", 2: "Enthusiasm", 3: "Neutral", 4: "Worry", 5: "Surprise", 
                       6: "Love Sentiment", 7: "Fun", 8: "Hate", 9: "Happiness", 10: "Boredom", 11: "Relief", 
                       12: "Anger"}

    predicted_emotion = emotion_mapping[np.argmax(prediction)]  # Convert prediction to emotion label

    # Render the prediction result in the HTML template
    return render_template("index.html", prediction_text=f"The emotion behind the text is {predicted_emotion}") 

if __name__ == "__main__":
    app.run(debug=True)
