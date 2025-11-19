from ml.tfidf_pipeline import TfidfPipeline

def test_fit_predict_basic():
    model = TfidfPipeline()
    texts = ["starbucks coffee", "mcdonalds burger"]
    labels = ["Coffee", "FastFood"]

    model.fit(texts, labels)
    out = model.predict(["starbucks latte"])[0]

    assert out["label"] in ["Coffee", "FastFood"]

def test_partial_fit_updates_label():
    model = TfidfPipeline()
    model.fit(["dummy"], ["A"])

    before = model.predict(["xyz"])[0]["label"]

    model.partial_fit(["xyz"], ["B"])
    after = model.predict(["xyz"])[0]["label"]

    assert after == "B"
