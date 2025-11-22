def fuse(rule_output, ml_output, tfidf_output, weights=None):
    # Example weighting and thresholds
    weights = weights or {"rule": 0.6, "ml": 0.3, "tfidf": 0.1}
    RULE_HIGH_CONF = 0.9
    TFIDF_HIGH_CONF = 0.10  # Lowered for demo: TF-IDF wins more easily after retrain

    # 1. Prefer rule if highly confident
    if rule_output and rule_output.get("confidence", 0) >= RULE_HIGH_CONF:
        final = rule_output.copy()
        final["model_used"] = "rule"
        final["rationale"] = {
            "rule_hits": rule_output.get("matches", []),
            "top_tokens": ml_output.get("top_tokens", []) if ml_output else [],
            "weighting": weights,
            "notes": "Rule wins with high confidence."
        }
        return final

    # 2. If TF-IDF is highly confident (post-correction/retrain), prefer it
    if tfidf_output and tfidf_output.get("confidence", 0) >= TFIDF_HIGH_CONF:
        final = tfidf_output.copy()
        final["model_used"] = "tfidf"
        final["rationale"] = {
            "rule_hits": rule_output.get("matches", []) if rule_output else [],
            "top_tokens": tfidf_output.get("top_tokens", []),
            "weighting": weights,
            "notes": "TF-IDF override (confidence due to recent feedback)."
        }
        return final

    # 3. Otherwise, fallback to ML (DistilBERT)
    if ml_output:
        final = ml_output.copy()
        final["model_used"] = "distilbert"
        final["rationale"] = {
            "rule_hits": rule_output.get("matches", []) if rule_output else [],
            "top_tokens": ml_output.get("top_tokens", []),
            "weighting": weights,
            "notes": "DistilBERT used: no strong rule or TF-IDF match."
        }
        return final

    # 4. Fallback to rule_output or unknown label
    return rule_output or {"label": "Unknown", "confidence": 0.0, "rationale": {}, "model_used": "none"}
