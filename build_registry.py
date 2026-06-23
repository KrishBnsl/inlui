import pandas as pd
from pathlib import Path
from datetime import datetime

def main():
    reports_dir = Path("models/reports")
    df_bench = pd.read_csv(reports_dir / "regression_benchmark.csv")
    
    # We select the top 3 models
    top_models = df_bench.head(3).copy()
    
    registry = []
    for idx, row in top_models.iterrows():
        registry.append({
            "Model_ID": f"REG_{datetime.now().strftime('%Y%m%d')}_{row['Model'].upper()}",
            "Type": "Regression",
            "Target": "closing_rank",
            "Train_Date": datetime.now().strftime('%Y-%m-%d'),
            "Test_MAE": round(row['test_MAE'], 2),
            "Test_RMSE": round(row['test_RMSE'], 2),
            "Test_R2": round(row['test_R2'], 4),
            "Hyperparameters": "alpha=10.0" if row['Model'] == 'Ridge' else "default",
            "Status": "Production" if row['Model'] == 'Ridge' else "Archived"
        })
        
    df_reg = pd.DataFrame(registry)
    df_reg.to_csv("models/model_registry.csv", index=False)
    print("model_registry.csv created.")

if __name__ == "__main__":
    main()
