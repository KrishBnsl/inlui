import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

class JoSAAPredictor:
    """Historical predictor retained for inspection; not the serving contract."""

    def __init__(self, artifact_dir: Path, data_dir: Path):
        self.artifact_dir = artifact_dir
        self.data_dir = data_dir
        self.load_artifacts()
        
    def load_artifacts(self):
        log.info("Loading artifacts for inference...")
        model_dict = joblib.load(self.artifact_dir / "final_regression_model.joblib")
        self.model = model_dict["model"] if isinstance(model_dict, dict) else model_dict
        self.prep_pipeline = joblib.load(self.artifact_dir / "preprocessing_pipeline.joblib")
        
        with open(self.artifact_dir / "label_encoders.json", "r") as f:
            self.encoders = json.load(f)
            
        # We need the empirical std_dev map to inject uncertainty
        # Instead of recalculating, we assume a simplified map or recalculate from residual test
        # For a robust inference, we should calculate it or use a default
        self.std_dev_default = 150.0 
        
        # Load the "Universe" of choices (using 2024 round 6 data as a realistic catalog)
        df_clean = pd.read_csv(self.data_dir / "processed" / "josaa_cleaned.csv")
        self.universe = df_clean[(df_clean["year"] == 2024) & (df_clean["round"] == 6)].copy()
        
        # Remove duplicates of program/institute combos
        self.universe = self.universe.drop_duplicates(subset=["institute", "program", "category", "gender", "quota"])

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded = df.copy()
        for col, mapping in self.encoders.items():
            if f"{col}_encoded" in mapping: # Just safety
                pass
            # Create _encoded column
            target_col = f"{col}_encoded" if col != "seat_pool" else "seat_pool_encoded"
            source_col = col
            
            if source_col in encoded.columns:
                # Apply mapping
                encoded[target_col] = encoded[source_col].map(mapping).fillna(-1)
                
        # Required features for the model
        req_features = [
            "year", "round", "institute_type_encoded", "institute_encoded",
            "program_encoded", "category_encoded", "quota_encoded", "gender_encoded",
            "opening_rank", "seat_pool_encoded"
        ]
        
        # Fill missing numeric
        if "opening_rank" not in encoded.columns:
            encoded["opening_rank"] = 0 # Fallback
            
        if "seat_pool_encoded" not in encoded.columns:
            encoded["seat_pool_encoded"] = 0
            
        return encoded[req_features]

    def _generate_explanation(self, row, student_rank: int) -> str:
        margin = row["predicted_closing_rank"] - student_rank
        prob = row["admission_probability"]
        
        if prob >= 0.8:
            cls = "safe"
            reason = f"the predicted cutoff ({int(row['predicted_closing_rank'])}) offers a comfortable safety margin of {int(margin)} ranks."
        elif prob >= 0.4:
            cls = "moderate"
            reason = "your rank is close to the expected boundary. Volatility could swing the cutoff either way."
        else:
            cls = "ambitious"
            reason = f"the predicted cutoff is significantly higher (lower numerically) than your rank by {int(abs(margin))} spots."
            
        return f"{row['institute']} {row['program']} is classified as a {cls} choice because {reason}"

    def predict(self, profile: dict) -> dict:
        rank = profile["rank"]
        category = profile["category"]
        gender = profile["gender"]
        pref_inst = profile.get("pref_inst", "")
        
        # Filter universe
        mask = (self.universe["category"] == category) & (self.universe["gender"] == gender)
        if pref_inst:
            mask = mask & (self.universe["institute_type"] == pref_inst)
            
        choices = self.universe[mask].copy()
        if len(choices) == 0:
            return {"error": "No matching programs found for this profile."}
            
        # We need opening_rank to feed the model. Since we are forecasting for next year,
        # we assume opening_rank ~ closing_rank of previous year / 1.5 as a rough proxy if missing
        # But our universe HAS opening_rank from 2024. We use that.
        
        features_df = self.encode_features(choices)
        
        # Inference
        X = self.prep_pipeline.transform(features_df)
        preds = self.model.predict(X)
        choices["predicted_closing_rank"] = preds
        
        # Monte Carlo
        n_sims = 1000
        simulated_cutoffs = np.random.normal(loc=preds, scale=self.std_dev_default, size=(n_sims, len(preds)))
        probs = np.mean(rank <= simulated_cutoffs, axis=0)
        p05 = np.percentile(simulated_cutoffs, 5, axis=0)
        p50 = np.percentile(simulated_cutoffs, 50, axis=0)
        p95 = np.percentile(simulated_cutoffs, 95, axis=0)
        
        choices["admission_probability"] = probs
        choices["ci_lower_5"] = np.maximum(1, p05)
        choices["expected_cutoff_50"] = p50
        choices["ci_upper_95"] = p95
        
        # Sort
        choices["margin_score"] = (choices["predicted_closing_rank"] - rank) / choices["predicted_closing_rank"]
        choices["score"] = (choices["admission_probability"] * 100) + (choices["margin_score"] * 20)
        choices = choices.sort_values(by="score", ascending=False)
        
        top_choices = choices.head(10)
        results = []
        for _, row in top_choices.iterrows():
            res = {
                "institute": row["institute"],
                "program": row["program"],
                "predicted_cutoff": int(row["predicted_closing_rank"]),
                "admission_probability": round(row["admission_probability"], 3),
                "uncertainty_interval": [int(row["ci_lower_5"]), int(row["ci_upper_95"])],
                "explanation": self._generate_explanation(row, rank)
            }
            results.append(res)
            
        return {
            "student_profile": profile,
            "recommended_colleges": results
        }

def main():
    raise SystemExit(
        "BLOCKED: predict.py uses the invalidated legacy estimator and heuristic "
        "uncertainty. Use the inference API with research/artifacts instead."
    )

if __name__ == "__main__":
    main()
