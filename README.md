# Speech Emotion Recognition (SER) using Keras, TensorFlow and Sklearn

This project focuses on **Speech Emotion Recognition (SER)**, where the goal was to identify human emotions from audio recordings. The project uses **Python**, **Keras**, **TensorFlow**, and **Scikit-learn** to train models that predict emotions such as happiness, sadness, anger, and calmness from audio inputs.

The project employs feature extraction techniques on audio data, followed by model training, and concludes with the deployment of a Flask API for real-time predictions. The complete implementation achieves a working model that is capable of classifying emotions from audio input with high accuracy.


## Project Objectives

Emotions play a critical role in human interaction. Recognizing emotions from speech is useful for entertainment, education, and healthcare applications. The main goal of this project is to **predict emotions from speech data** using machine learning.

Our **Secondary Goal** was to deploy the model as a Flask API to allow real-time interaction, where users can upload an audio file and receive an emotion classification.

## Project Overview

This project followed a structured approach:
1. **Data Preparation**: Loading the audio dataset and extracting relevant features.
2. **Feature Extraction**: Leveraging audio processing techniques such as MFCC and Chroma features.
3. **Modeling**: Implementing a machine learning model (MLP) and a deep learning model (ANN) to classify emotions.
4. **Evaluation**: Testing and evaluating the model's performance with metrics like accuracy.
5. **API Deployment**: Creating a Flask API for real-time emotion prediction from audio files.

## Dataset

We used the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**, a publicly available dataset containing emotional speech and song. The dataset consists of two lexically-matched statements,  vocalized in a neutral North American accent by 24 professional actors (12 female, 12 male). The speech files portray different emotions, including:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

Datset Link: https://zenodo.org/records/1188976

The dataset contains **7356 audio files**, and we focus on this project's **speech data**.

## Technologies Used

The project uses the following tech stack:
- **Programming Language**: Python
- **Machine Learning**: TensorFlow, Keras, Scikit-learn
- **Audio Processing**: Librosa, SoundFile
- **Data Handling**: Pandas, Numpy
- **Visualization**: Matplotlib

## EDA
![](/assets/Chroma Spectogram.png)


## Deployment

To make the model accessible, we developed a Flask API that allows users to upload an audio file and receive an emotion prediction. The API processes the audio file and feeds the features to the pre-trained model for classification.

#### Key Features of the API:
- **/predict** endpoint: Accepts an audio file in WAV format, processes it, and returns the predicted emotion.
- **Real-time Interaction**: The API is designed for real-time emotion prediction, making it usable for applications such as sentiment analysis or interactive voice systems.

The API was containerized using Docker for easier deployment. A `Dockerfile` was created to set up the Flask environment, install dependencies, and expose the necessary port for the API.

## Achievements
By the end of the project, the following goals were successfully achieved:
- Built a working speech emotion recognition model that can accurately classify emotions.
- Extracted relevant audio features using advanced signal processing techniques.
- Deployed the model using a Flask API, making it accessible for real-time predictions.
- Dockerized the application for easier deployment on cloud platforms like Heroku or AWS.
