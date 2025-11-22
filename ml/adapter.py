# ml/adapter.py
# -----------------------------
# Backend-facing interface for prediction and full retraining.
# Updated to use database instead of JSON file for consistency with backend.

from ml.tfidf_pipeline import TfidfPipeline
from ml.data_pipeline import normalize_text
import pandas as pd
import json
import os
import sys

MODEL_DIR = "saved_models/tfidf"
BASE_DATASET = "data/train.csv"

_model = None


def load_model():
    """Load the TF-IDF model once and keep it cached."""
    global _model
    if _model is None:
        _model = TfidfPipeline()
        _model.load(MODEL_DIR)
    return _model


def predict_text(text: str):
    """Normalize text → run prediction → return model output dict."""
    model = load_model()
    cleaned = normalize_text(text)
    return model.predict([cleaned])[0]


def retrain_from_feedback():
    """
    FULL retrain using:
    - Original dataset
    - Feedback data from database (consistent with backend)
    This guarantees stable learning and correct demo behavior.
    """
    if not os.path.exists(BASE_DATASET):
        return {"status": "base_dataset_missing"}

    base_df = pd.read_csv(BASE_DATASET)

    # Get feedback from database (same as backend)
    try:
        # Import backend storage to get feedback from database
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from backend.storage import get_feedback_samples
        
        samples = get_feedback_samples()
        feedback_count = len(samples)
        
        # Combine base dataset with feedback
        if feedback_count > 0:
            # Create feedback DataFrame from database
            feedback_data = [(text, label) for _, text, label in samples]
            fb_df = pd.DataFrame(feedback_data, columns=["transaction_text", "category"])
            combined = pd.concat([base_df, fb_df], ignore_index=True)
        else:
            # No feedback, just use base dataset
            combined = base_df
            feedback_count = 0
    except Exception as e:
        # Fallback: try to read from JSON file if database not available (for tests)
        try:
            FEEDBACK_PATH = "data/feedback.json"
            feedback = json.load(open(FEEDBACK_PATH))
            if feedback:
                fb_df = pd.DataFrame(feedback)
                # Handle both old format (dict with keys) and new format (list of lists)
                if isinstance(feedback[0], dict):
                    if "text" in feedback[0] and "correct_label" in feedback[0]:
                        fb_df = pd.DataFrame([{"transaction_text": f["text"], "category": f["correct_label"]} for f in feedback])
                    else:
                        fb_df.columns = ["transaction_text", "category"]
                else:
                    fb_df.columns = ["transaction_text", "category"]
                combined = pd.concat([base_df, fb_df], ignore_index=True)
                feedback_count = len(feedback)
            else:
                combined = base_df
                feedback_count = 0
        except:
            # If both database and JSON fail, just use base dataset
            combined = base_df
            feedback_count = 0

    # FULL model rebuild
    model = TfidfPipeline()
    texts = combined["transaction_text"].map(normalize_text).tolist()
    labels = combined["category"].tolist()

    model.fit(texts, labels)
    model.save(MODEL_DIR)

    # Reload fresh model into memory
    global _model
    _model = model

    return {
        "status": "retrained_fully",
        "feedback_used": feedback_count
    }