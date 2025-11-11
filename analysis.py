import pandas as pd
import numpy as np
import re
import warnings
import torch
from tabpfn import TabPFNClassifier

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set PyTorch to single thread for stability
torch.set_num_threads(1)

# ========================================
# 1. Dummy Data Creation
# ========================================

def create_dummy_data():
    """
    Creates a simulated dataset mimicking the study's structure.
    - 'cancer_type': Simulates different rare cancers.
    - 'split': Simulates the geographic dev/test split [cite: 9, 51-52].
    - 'target': Simulates 5-year mortality (1=Dead, 0=Survived)[cite: 48].
    - Features: Mix of categorical and numerical.
    """
    print("Creating dummy data...")
    
    def generate_data(n, cancer_name, split):
        return {
            'Age': np.random.randint(20, 90, n),
            'Sex': np.random.choice(['Male', 'Female'], n),
            'Stage': np.random.choice(['I', 'II', 'III', 'IV', 'Unknown'], n, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
            'Income': np.random.normal(50000, 15000, n).astype(int),
            'target': np.random.choice([0, 1], n, p=[0.6, 0.4]),
            'cancer_type': cancer_name,
            'split': split,
        }

    # Cancer A: Decent size
    df_a_dev = pd.DataFrame(generate_data(800, 'Cancer A', 'dev'))
    df_a_test = pd.DataFrame(generate_data(200, 'Cancer A', 'test'))
    
    # Cancer B: Medium size
    df_b_dev = pd.DataFrame(generate_data(300, 'Cancer B', 'dev'))
    df_b_test = pd.DataFrame(generate_data(80, 'Cancer B', 'test'))
    
    # Cancer C: Very small size (to test adaptive CV)
    df_c_dev = pd.DataFrame(generate_data(40, 'Cancer C', 'dev'))
    df_c_test = pd.DataFrame(generate_data(15, 'Cancer C', 'test'))
    
    # Ensure at least one positive/negative class in small cohort
    df_c_dev.loc[0, 'target'] = 0
    df_c_dev.loc[1, 'target'] = 1
    df_c_test.loc[0, 'target'] = 0
    df_c_test.loc[1, 'target'] = 1

    df = pd.concat([df_a_dev, df_a_test, df_b_dev, df_b_test, df_c_dev, df_c_test], ignore_index=True)
    
    print(f"Dummy data created with {len(df)} total rows.")
    print("Cancer type distribution:")
    print(df.groupby(['cancer_type', 'split']).size().unstack())
    print("-" * 60)
    return df

# ========================================
# 2. Model Factory Functions
# ========================================

def create_tabpfn():
    """Factory for TabPFNClassifier with CPU-safe settings"""
    return TabPFNClassifier(
        device="cpu",
        random_state=42,
    )

def create_logistic_regression():
    """Factory for Logistic Regression"""
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )

def create_random_forest():
    """Factory for Random Forest"""
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

def create_xgboost():
    """Factory for XGBoost"""
    return xgb.XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        enable_categorical=False,
        verbosity=0
    )

# Model configuration dictionary
model_factories = {
    "TabPFN": create_tabpfn,
    "LogisticRegression": create_logistic_regression,
    "RandomForest": create_random_forest,
    "XGBoost": create_xgboost,
}

# ========================================
# 3. Helper Functions
# ========================================

def sanitize_column_names(df):
    """Sanitizes column names for XGBoost compatibility."""
    new_columns = {}
    for col in df.columns:
        sanitized = str(col)
        sanitized = re.sub(r'[<>\[\](),\-\s]+', '_', sanitized)
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        new_columns[col] = sanitized
    return df.rename(columns=new_columns)

def setup_adaptive_cv(X_dev, y_dev, min_samples_per_fold=1, max_folds=5):
    """
    Selects a CV strategy based on sample size and class distribution.
    """
    class_counts = np.bincount(y_dev)
    min_class_size = class_counts.min()
    
    # Calculate optimal folds, ensuring at least 2
    optimal_folds = min_class_size // min_samples_per_fold
    n_splits = max(2, min(max_folds, optimal_folds))
    
    print(f"    [CV Setup] Min class size: {min_class_size}. Using n_splits={n_splits}")

    try:
        if n_splits >= 3:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_splits = list(skf.split(X_dev, y_dev))
            cv_method = f"StratifiedKFold(n_splits={n_splits})"
        else:
            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)
            cv_splits = list(sss.split(X_dev, y_dev))
            cv_method = "StratifiedShuffleSplit"
            
    except ValueError:
        train_idx, valid_idx = train_test_split(
            range(len(X_dev)), test_size=0.3, stratify=y_dev, random_state=42
        )
        cv_splits = [(train_idx, valid_idx)]
        n_splits = 1
        cv_method = "Single train-validation split (fallback)"
    
    print(f"    [CV Setup] Method: {cv_method}")
    return cv_splits, n_splits

# ========================================
# 4. Main Analysis Loop
# ========================================

def main():
    """
    Main function to run the conceptual analysis.
    """
    
    # 1. Load Data (using dummy data here)
    df = create_dummy_data()
    
    # Get list of "cancers" to process
    valid_cancers = df['cancer_type'].unique()
    
    overall_metrics = []

    # 2. Loop over each cancer type
    for cancer_name in valid_cancers:
        print(f"\n{'='*60}")
        print(f"Processing Cancer Type: {cancer_name}")
        print(f"{'='*60}")
        
        df_cancer = df[df["cancer_type"] == cancer_name].copy()
        
        # 3. Create Development and Test Sets
        df_dev = df_cancer[df_cancer["split"] == "dev"]
        df_test = df_cancer[df_cancer["split"] == "test"]
        
        # Define features (X) and target (y)
        X_dev = df_dev.drop(columns=["target", "cancer_type", "split"])
        y_dev = df_dev["target"].values
        X_test = df_test.drop(columns=["target", "cancer_type", "split"])
        y_test = df_test["target"].values
        
        print(f"  Data sizes - Dev: {len(X_dev)}, Test: {len(X_test)}")
        if len(X_dev) == 0 or len(X_test) == 0:
            print("  Skipping: Not enough dev or test data.")
            continue
            
        # 4. Preprocessing
        categorical_cols = X_dev.select_dtypes(include=["object"]).columns
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        
        if len(categorical_cols) > 0:
            print(f"  Encoding columns: {list(categorical_cols)}")
            X_dev[categorical_cols] = encoder.fit_transform(X_dev[categorical_cols])
            X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])
            
        # Sanitize column names for XGBoost
        X_dev = sanitize_column_names(X_dev)
        X_test = sanitize_column_names(X_test)

        # 5. Setup Adaptive Cross-Validation 
        try:
            cv_splits, n_splits = setup_adaptive_cv(X_dev, y_dev)
        except Exception as e:
            print(f"  Skipping {cancer_name}: Error setting up CV. {e}")
            continue

        # 6. Model Training and Evaluation Loop
        for model_name, model_factory in model_factories.items():
            print(f"\n  --- Training {model_name} ---")
            
            # Arrays to store predictions
            oof_preds_dev = np.zeros(len(X_dev))
            preds_test_folds = np.zeros((len(X_test), n_splits))
            
            for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits):
                X_train_fold = X_dev.iloc[train_idx]
                y_train_fold = y_dev[train_idx]
                X_valid_fold = X_dev.iloc[valid_idx]
                
                try:
                    model = model_factory()
                    model.fit(X_train_fold, y_train_fold)
                    
                    # Store OOF predictions for dev set
                    oof_preds_dev[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
                    
                    # Store test predictions from this fold
                    preds_test_folds[:, fold_idx] = model.predict_proba(X_test)[:, 1]

                except Exception as e:
                    print(f"    ERROR training {model_name} on fold {fold_idx+1}: {e}")
                    # On error, fill with 0.5 (neutral proba)
                    oof_preds_dev[valid_idx] = 0.5 
                    preds_test_folds[:, fold_idx] = 0.5
            
            # 7. Aggregate and Evaluate Results
            test_preds_ens = preds_test_folds.mean(axis=1)
            
            try:
                # Calculate AUC
                dev_auc = roc_auc_score(y_dev, oof_preds_dev)
                test_auc = roc_auc_score(y_test, test_preds_ens)
                
                print(f"    [{model_name}] Dev OOF AUC: {dev_auc:.4f}")
                print(f"    [{model_name}] Test (Ens) AUC: {test_auc:.4f}")
                
                overall_metrics.append({
                    "cancer_name": cancer_name,
                    "model": model_name,
                    "dev_auc": dev_auc,
                    "test_auc": test_auc,
                    "dev_samples": len(y_dev),
                    "test_samples": len(y_test),
                })
            except Exception as e:
                print(f"    [{model_name}] ERROR calculating AUC: {e}")

    # 8. Final Summary
    print(f"\n{'='*60}")
    print("Analysis Complete: Overall Results")
    print(f"{'='*60}")
    
    if overall_metrics:
        results_df = pd.DataFrame(overall_metrics)
        print(results_df.round(4))
        
        # Save results to CSV
        output_path = "analysis_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    else:
        print("No results were generated.")


if __name__ == "__main__":
    print("=" * 60)
    print("Analysis with TabPFN")
    print("=" * 60)
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print("=" * 60 + "\n")
    
    main()