import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
DATA_PATH = "C:/Users/sindhu/OneDrive/Desktop/pro/V2Emotion"
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load data from dataset folder
def load_dataset():
    X, y = [], []
    for actor in os.listdir(DATA_PATH):
        actor_folder = os.path.join(DATA_PATH, actor)
        if not os.path.isdir(actor_folder):
            continue
        for file in os.listdir(actor_folder):
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code = parts[2]
                emotion = emotion_map.get(emotion_code)
                file_path = os.path.join(actor_folder, file)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(emotion)
    return np.array(X), np.array(y)
if __name__ == "__main__":
    print("🔄 Loading dataset...")
    X, y = load_dataset()
    print(f"✅ Loaded {len(X)} audio samples.")
    print("📐 Feature vector shape:", X[0].shape if len(X) > 0 else "N/A")
    print("🎭 Emotions found:", sorted(set(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n🚀 Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
    joblib.dump(clf, "emotion_model.pkl")
    print("\n💾 Model saved as 'emotion_model.pkl'")
