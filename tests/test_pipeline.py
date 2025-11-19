from ml.data_pipeline import normalize_text

def test_normalize_removes_special_chars():
    assert normalize_text("STARBCKS #1!") == "starbcks 1"

def test_alias_substitution():
    assert "starbucks" in normalize_text("Starbcks Coffee")
