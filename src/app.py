from flask import Flask, request, jsonify
import joblib
import librosa
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('emotion_classifier.pkl')

# Define a feature extraction function
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request has an audio file
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Save the file temporarily
    file_path = './temp.wav'
    file.save(file_path)
    
    # Extract features from the audio file
    features = extract_features(file_path)
    
    # Reshape features for the model (if required)
    features = np.array([features])
    
    # Predict emotion
    prediction = model.predict(features)
    
    # Convert prediction to the actual emotion label (based on your model)
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    predicted_emotion = emotions[int(prediction[0])]
    
    # Return prediction as JSON
    return jsonify({'emotion': predicted_emotion})

if __name__ == '__main__':
    app.run(debug=True)

