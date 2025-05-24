# 🎤 Voice Emotion Detection using Machine Learning

This project implements a machine learning-based system for detecting emotions from voice recordings. By analyzing audio input, the system identifies emotional states such as happy, sad, angry, surprised, and more.

The core of the project lies in the extraction of audio features using Mel Frequency Cepstral Coefficients (MFCCs), which effectively capture the characteristics of human speech. These features are used to train a Random Forest Classifier that learns to associate speech patterns with specific emotions.

A simple and intuitive user interface is built using **Streamlit**, enabling users to upload audio files in various formats (WAV, MP3, etc.) and get instant emotion predictions.

---

## 🔍 Features

- Emotion classification using speech signals.
- MFCC-based feature extraction.
- Trained Random Forest model for robust prediction.
- User-friendly web app with file upload and emotion display.
- Supports multiple audio formats for wider usability.

---

## 🛠️ Technologies Used

- **Python** – Core programming language.
- **Librosa** – Audio processing and MFCC feature extraction.
- **Scikit-learn** – Machine learning (Random Forest, model evaluation).
- **Streamlit** – Interactive web interface.
- **Pydub** – Audio format conversion.
- **Joblib** – Model saving and loading.

---

## 📈 Project Workflow

1. Audio data is collected from a labeled dataset.
2. MFCC features are extracted from each audio file.
3. A Random Forest classifier is trained using these features.
4. The model is saved for future predictions.
5. A web interface is created to allow users to upload audio and view detected emotions in real-time.


