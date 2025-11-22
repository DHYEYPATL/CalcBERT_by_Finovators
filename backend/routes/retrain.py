"""
Retrain endpoint - triggers full model retraining for demo.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.storage import get_feedback_samples
from backend.config import settings

router = APIRouter()

class RetrainRequest(BaseModel):
    """Request schema for retrain endpoint."""
    mode: Literal["incremental", "full"] = Field(
        default="full",
        description="Retrain mode: incremental or full"
    )
    model: Literal["tfidf", "distilbert"] = Field(
        default="tfidf",
        description="Model to retrain: tfidf or distilbert"
    )
    class Config:
        json_schema_extra = {
            "example": {
                "mode": "full",
                "model": "tfidf"
            }
        }

class RetrainResponse(BaseModel):
    """Response schema for retrain endpoint."""
    status: str = Field(..., description="Status: started, complete, or error")
    details: str = Field(..., description="Details about the retrain operation")
    samples_used: int = Field(default=0, description="Number of feedback samples used")

def _run_full_tfidf() -> dict:
    """
    Run FULL TF-IDF retraining using BASE DATASET + ALL feedback.
    This ensures the model learns from both original data and user corrections.
    Returns:
        Dictionary with retrain results.
    """
    try:
        from ml.tfidf_pipeline import TfidfPipeline
        from ml.data_pipeline import normalize_text
        import pandas as pd
        import os
        
        BASE_DATASET = "data/train.csv"  # Updated to match ml/adapter.py
        
        # Load base dataset
        if not os.path.exists(BASE_DATASET):
            return {
                "status": "error",
                "details": f"Base dataset not found at {BASE_DATASET}",
                "samples_used": 0
            }
        
        base_df = pd.read_csv(BASE_DATASET)
        
        # Get feedback from database
        samples = get_feedback_samples()
        feedback_count = len(samples)
        
        # Combine base dataset with feedback
        if feedback_count > 0:
            # Create feedback DataFrame
            feedback_data = [(text, label) for _, text, label in samples]
            fb_df = pd.DataFrame(feedback_data, columns=["transaction_text", "category"])
            # Combine with base dataset
            combined_df = pd.concat([base_df, fb_df], ignore_index=True)
        else:
            # No feedback, just use base dataset
            combined_df = base_df
        
        # Normalize texts and prepare for training
        texts = combined_df["transaction_text"].map(normalize_text).tolist()
        labels = combined_df["category"].tolist()
        
        # Check unique categories before training
        unique_categories = sorted(set(labels))
        base_categories = sorted(set(base_df["category"].unique()))
        
        # Verify we have all base categories
        if len(unique_categories) < len(base_categories):
            print(f"WARNING: Combined dataset has {len(unique_categories)} categories but base had {len(base_categories)}")
            print(f"Base categories: {base_categories}")
            print(f"Combined categories: {unique_categories}")
        
        # FULL retrain on combined dataset
        pipeline = TfidfPipeline()
        pipeline.fit(texts, labels)   # FULL fit on all data!
        pipeline.save(settings.TFIDF_MODEL_DIR)
        
        # Verify categories after training
        trained_categories = sorted([str(c) for c in pipeline.le.classes_])
        
        # Double-check: reload the saved model to verify it was saved correctly
        verify_pipeline = TfidfPipeline()
        verify_pipeline.load(settings.TFIDF_MODEL_DIR)
        verified_categories = sorted([str(c) for c in verify_pipeline.le.classes_])
        
        if len(verified_categories) != len(trained_categories):
            print(f"ERROR: Model save/load mismatch! Saved {len(trained_categories)} but loaded {len(verified_categories)}")
        
        return {
            "status": "complete",
            "details": f"Full TF-IDF retrain completed: {len(base_df)} base samples + {feedback_count} feedback samples = {len(combined_df)} total. Categories: {len(verified_categories)} (base had {len(base_categories)}, expected 8)",
            "samples_used": len(combined_df),
            "categories_trained": len(verified_categories),
            "categories_list": verified_categories,
            "base_categories_count": len(base_categories)
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "details": f"Retrain failed: {str(e)}\n{traceback.format_exc()}",
            "samples_used": 0
        }

@router.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest, background_tasks: BackgroundTasks) -> RetrainResponse:
    """
    Trigger full model retraining (TF-IDF only, for demo).
    After retraining, reloads the model in ModelAdapter so predictions use the updated model.
    """
    if req.model != "tfidf":
        raise HTTPException(
            status_code=400,
            detail="Only TF-IDF full retrain is supported via API for this demo."
        )
    # For demo: ignore 'mode', always do full retrain.
    if settings.RETRAIN_SYNC:
        try:
            # Run full retrain
            result = _run_full_tfidf()
            
            # Reload the model in ModelAdapter so predictions use the updated model
            try:
                from backend.routes.predict import adapter
                if adapter is not None:
                    reload_success = adapter.reload_tfidf_model()
                    if reload_success:
                        result["details"] += " | Model reloaded in memory."
                    else:
                        result["details"] += " | WARNING: Model reload failed - restart backend to use new model."
                else:
                    result["details"] += " | WARNING: Adapter not available - restart backend to use new model."
            except Exception as e:
                result["details"] += f" | WARNING: Could not reload model: {str(e)} - restart backend to use new model."
            
            return RetrainResponse(**result)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Retrain failed: {str(e)}",
            )
    else:
        background_tasks.add_task(_run_full_tfidf)
        return RetrainResponse(
            status="started",
            details="Full TF-IDF retrain started in background",
            samples_used=0
        )

@router.get("/retrain/status")
def get_retrain_status() -> dict:
    """
    Get retrain configuration and status.
    Returns:
        Dictionary with retrain configuration
    """
    return {
        "sync_mode": settings.RETRAIN_SYNC,
        "supported_models": ["tfidf"],
        "supported_modes": ["full"],
        "message": "Retrain endpoint is ready!"
    }
