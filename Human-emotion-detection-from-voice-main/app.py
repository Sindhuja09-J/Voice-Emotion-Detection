import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os
from pydub import AudioSegment

# Load the trained model once
@st.cache_data
def load_model():
    return joblib.load("emotion_model.pkl")
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs.reshape(1, -1)
def convert_to_wav(uploaded_file):
    if uploaded_file.type == "audio/wav":
        return uploaded_file
    else:
        audio = AudioSegment.from_file(uploaded_file)
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(tmp_wav.name, format="wav")
        return tmp_wav

def main():
    st.title(" Voice Emotion Detection")
    st.write("Upload or record your voice ")

    model = load_model()

    uploaded_file = st.file_uploader("Upload an audio file (wav, mp3, etc.)", type=["wav", "mp3", "ogg", "flac"])
    
    if uploaded_file is not None:
        # Convert if not wav
        if uploaded_file.type != "audio/wav":
            tmp_file = convert_to_wav(uploaded_file)
            file_path = tmp_file.name
        else:
            file_path = uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

        st.audio(uploaded_file)

        features = extract_features(file_path)
        prediction = model.predict(features)[0]
        st.markdown(f"### Predicted Emotion: **{prediction.upper()}**")

        # Clean up temp file if created
        if uploaded_file.type != "audio/wav":
            os.unlink(tmp_file.name)

    else:
        st.info("Please upload an audio file ")

if __name__ == "__main__":
    main()
