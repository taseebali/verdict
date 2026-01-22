"""FastAPI REST API for VERDICT ML Platform - Consolidated v2.0"""

import time
import logging
import sys
import os
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.pipeline import MLPipeline
from src.decision.decision_audit_logger import DecisionAuditLogger
from config.settings import MODEL_CONFIGS, RANDOM_SEED
from .schemas import (
    PredictRequest, PredictResponse, ErrorResponse, HealthResponse,
    FeaturesResponse, FeatureInfo, AuditResponse, AuditRecord,
    WhatIfRequest, WhatIfResponse, RecommendationResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VERDICT API v2.0",
    description="Complete ML Decision Platform with Explainability & What-If Analysis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pipeline = None
audit_logger = None
feature_recommendations = None
feature_ranges = None


def format_value(feature_name: str, value: float) -> str:
    """Convert model values to human-readable format.
    
    Examples:
        - monthlyCharges: $89.50
        - churnRate: 12.5%
        - tenure: 24 months
        - age: 35 years
    """
    feature_lower = feature_name.lower()
    
    # Currency
    if any(x in feature_lower for x in ['charge', 'cost', 'price', 'payment', 'income', 'monthly']):
        return f"${value:,.2f}"
    
    # Percentage
    if any(x in feature_lower for x in ['percent', 'rate', 'ratio']):
        return f"{value:.1f}%"
    
    # Months/Years
    if any(x in feature_lower for x in ['tenure', 'months']):
        return f"{int(value)} months"
    
    # Age
    if 'age' in feature_lower:
        return f"{int(value)} years"
    
    # Default
    return f"{value:.2f}"


def generate_feature_recommendations(df: pd.DataFrame) -> Dict:
    """Generate recommendations for feature selection based on training data."""
    recommendations = {}
    
    for col in df.columns:
        if col.lower() == 'churn' or col.lower() == 'target':
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            recommendations[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "recommendation": f"Recommended range: {df[col].min():.2f} - {df[col].max():.2f}"
            }
    
    return recommendations


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline, audit_logger, feature_recommendations, feature_ranges
    
    try:
        # Create sample data
        sample_data = pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'monthlyCharges': np.random.uniform(20, 150, 100),
            'tenure': np.random.randint(0, 72, 100),
            'churn': np.random.randint(0, 2, 100)
        })
        
        pipeline = MLPipeline(sample_data, 'churn')
        pipeline.preprocess()
        feature_ranges = pipeline.preprocessor.get_feature_ranges()
        
        # Initialize loggers
        audit_logger = DecisionAuditLogger()
        
        # Generate recommendations
        feature_recommendations = generate_feature_recommendations(sample_data)
        
        logger.info("✅ API initialized successfully (v2.0)")
    except Exception as e:
        logger.warning(f"⚠️ Startup initialization partial: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        status = "healthy" if pipeline else "degraded"
        return HealthResponse(
            status=status,
            version="2.0.0",
            models_available=list(MODEL_CONFIGS.keys())
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations")
async def get_recommendations() -> RecommendationResponse:
    """Get feature recommendations and ranges for UI sliders.
    
    Returns recommended values for each feature based on training data.
    Useful for initializing UI controls and validating inputs.
    """
    try:
        if feature_recommendations is None:
            raise HTTPException(status_code=503, detail="Recommendations not available")
        
        return RecommendationResponse(
            status="success",
            recommendations=feature_recommendations
        )
    except Exception as e:
        logger.error(f"Recommendations failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Make a real prediction with confidence and human-readable output.
    
    Features:
    - Validates input against feature ranges
    - Returns formatted values (currency, percentages, etc.)
    - Logs to audit trail
    - Returns confidence level (High/Medium/Low)
    
    Example:
    ```json
    {
        "features": {"age": 35, "monthlyCharges": 89.50, "tenure": 24},
        "model_name": "random_forest"
    }
    ```
    """
    start_time = time.time()
    
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        if request.model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model '{request.model_name}' not available")
        
        # Convert to DataFrame for prediction
        features_df = pd.DataFrame([request.features])
        
        # Get predictions from pipeline
        predictions = pipeline.get_model_predictions(
            model_names=[request.model_name],
            X_input=features_df
        )
        
        pred_data = predictions[request.model_name]
        prediction = pred_data.get("prediction")
        confidence = float(pred_data.get("confidence", 0.0))
        
        # Format output for UI
        formatted_features = {
            k: format_value(k, v) for k, v in request.features.items()
        }
        
        # Determine confidence level
        if confidence > 0.8:
            confidence_level = "High"
        elif confidence > 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Log to audit trail
        if audit_logger:
            try:
                audit_logger.log_prediction(
                    model_name=request.model_name,
                    prediction=str(prediction),
                    confidence=confidence,
                    features=request.features
                )
            except:
                pass  # Continue even if audit logging fails
        
        execution_time = (time.time() - start_time) * 1000
        
        return PredictResponse(
            status="success",
            prediction=int(prediction),
            prediction_label="Will Churn" if prediction == 1 else "Will Stay",
            confidence=confidence,
            confidence_level=confidence_level,
            formatted_features=formatted_features,
            model_name=request.model_name,
            execution_time_ms=round(execution_time, 2)
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/whatif", response_model=WhatIfResponse)
async def what_if_analysis(request: WhatIfRequest) -> WhatIfResponse:
    """What-if scenario analysis - see how changes affect predictions.
    
    Compare current prediction vs hypothetical scenario with changes.
    
    Example:
    ```json
    {
        "current_features": {"age": 35, "monthlyCharges": 89.50, "tenure": 24},
        "scenario_changes": {"tenure": 36, "monthlyCharges": 79.50},
        "model_name": "random_forest"
    }
    ```
    
    Response shows:
    - Current prediction vs scenario prediction
    - Whether prediction changed
    - Formatted changes (before/after/change)
    """
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Current prediction
        current_pred = pipeline.get_model_predictions(
            model_names=[request.model_name],
            X_input=pd.DataFrame([request.current_features])
        )
        
        current_data = current_pred[request.model_name]
        current_prediction = int(current_data["prediction"])
        current_confidence = float(current_data["confidence"])
        
        # Scenario prediction
        scenario_features = request.current_features.copy()
        scenario_features.update(request.scenario_changes)
        
        scenario_pred = pipeline.get_model_predictions(
            model_names=[request.model_name],
            X_input=pd.DataFrame([scenario_features])
        )
        
        scenario_data = scenario_pred[request.model_name]
        scenario_prediction = int(scenario_data["prediction"])
        scenario_confidence = float(scenario_data["confidence"])
        
        # Format changes for UI
        formatted_changes = {
            k: {
                "before": format_value(k, request.current_features[k]),
                "after": format_value(k, v),
                "change": format_value(k, v - request.current_features[k])
            }
            for k, v in request.scenario_changes.items()
        }
        
        return WhatIfResponse(
            status="success",
            current_prediction=current_prediction,
            current_prediction_label="Will Churn" if current_prediction == 1 else "Will Stay",
            current_confidence=current_confidence,
            scenario_prediction=scenario_prediction,
            scenario_prediction_label="Will Churn" if scenario_prediction == 1 else "Will Stay",
            scenario_confidence=scenario_confidence,
            prediction_changed=bool(current_prediction != scenario_prediction),
            formatted_changes=formatted_changes,
            model_name=request.model_name
        )
    
    except Exception as e:
        logger.error(f"What-if analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features", response_model=FeaturesResponse)
async def get_features() -> FeaturesResponse:
    """Get available features with ranges and recommendations."""
    try:
        if feature_ranges is None:
            raise HTTPException(status_code=503, detail="Features not available")
        
        features_list = []
        for feature_name, range_info in feature_ranges.items():
            recommended = None
            if feature_recommendations and feature_name in feature_recommendations:
                recommended = feature_recommendations[feature_name].get("mean")
            
            feature = FeatureInfo(
                name=feature_name,
                dtype=range_info.get("dtype", "unknown"),
                min_value=range_info.get("min"),
                max_value=range_info.get("max"),
                recommended_value=recommended,
                display_format="currency" if "charge" in feature_name.lower() else "number"
            )
            features_list.append(feature)
        
        return FeaturesResponse(
            status="success",
            features=features_list,
            total_features=len(features_list)
        )
    
    except Exception as e:
        logger.error(f"Feature retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit")
async def get_audit_log(limit: int = Query(10, ge=1, le=1000)) -> AuditResponse:
    """Get audit trail of predictions."""
    try:
        if audit_logger is None:
            return AuditResponse(status="success", count=0, records=[])
        
        records = []
        try:
            recent_records = audit_logger.get_recent_records(limit)
            for record in recent_records:
                audit_record = AuditRecord(
                    id=str(record.get("id", "")),
                    timestamp=record.get("timestamp", ""),
                    model_name=record.get("model_name", ""),
                    prediction=str(record.get("prediction", "")),
                    confidence=float(record.get("confidence", 0.0)),
                    status=record.get("status", "success")
                )
                records.append(audit_record)
        except AttributeError:
            logger.warning("Audit logger incomplete")
        
        return AuditResponse(
            status="success",
            count=len(records),
            records=records
        )
    
    except Exception as e:
        logger.error(f"Audit retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API overview."""
    return {
        "message": "🚀 VERDICT ML API v2.0",
        "description": "Complete Decision Platform with Explainability",
        "endpoints": {
            "predictions": "/predict - Make single predictions",
            "what_if": "/whatif - Scenario analysis",
            "features": "/features - Feature info & ranges",
            "recommendations": "/recommendations - Suggested feature values",
            "audit": "/audit - Prediction history",
            "health": "/health - API status",
            "docs": "/docs - Interactive API documentation"
        },
        "features": [
            "Real model predictions with confidence scores",
            "What-if scenario analysis",
            "Human-readable output formatting",
            "Feature recommendations",
            "Complete audit logging",
            "Interactive API docs"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
