import json
from ml.tfidf_pipeline import TfidfPipeline
from ml.data_pipeline import normalize_text

def ingest_feedback(path="data/feedback.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except:
        return []

    samples = []
    for item in data:
        normalized = normalize_text(item["text"])
        samples.append((normalized, item["correct_label"]))
    return samples

def apply_incremental_update(pipeline: TfidfPipeline, samples, save_dir="saved_models/tfidf"):
    if not samples:
        return 0

    texts, labels = zip(*samples)
    
    updated_count = pipeline.partial_fit(list(texts), list(labels))
    if updated_count > 0:
        pipeline.save(save_dir)
    
    return updated_count



