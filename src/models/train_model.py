import matplotlib.pyplot as plt
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def train_model(pipeline, x_train, y_train, x_test, y_test, experiment_name="Customer_Churn_Experiment"):
    """
    Train model inside pipeline, log results with MLflow, and save model locally.
    """

    # ایجاد یا انتخاب experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="XGBClassifier_Run"):

        # --- Training ---
        pipeline.fit(x_train, y_train)

        # --- Evaluation ---
        y_pred = pipeline.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        # --- Log metrics ---
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # --- Extract features for importance plot ---
        preprocessor = pipeline.named_steps['preprocessor']
        num_features = preprocessor.transformers_[0][2]
        cat_encoder = preprocessor.transformers_[1][1]
        cat_features = preprocessor.transformers_[1][2]

        if hasattr(cat_encoder, 'get_feature_names_out'):
            cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
        else:
            cat_feature_names = cat_encoder.get_feature_names(cat_features)

        all_features = list(num_features) + list(cat_feature_names)

        # Feature importances
        model = pipeline.named_steps['model']
        feature_importances = pd.Series(model.feature_importances_, index=all_features)

        # --- Plot feature importance ---
        plt.figure(figsize=(12, 5))
        feature_importances.sort_values(ascending=True).plot(kind='barh', color='skyblue')
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Feature')
        plt.title(f'Feature Importances - {type(model).__name__}')
        plt.tight_layout()

        # Save plot
        plot_path = "image/feature_importance.png"
        os.makedirs("image", exist_ok=True)
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)  

        # --- Save model locally ---
        save_path = "src/models"
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, f"{type(model).__name__}.joblib")
        joblib.dump(pipeline, model_path)

        # --- Log model in MLflow ---
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"✅ Model trained and saved at {model_path}")
        print(f"✅ Metrics logged: Accuracy={acc:.4f}, F1={f1:.4f}, ROC_AUC={roc_auc:.4f}")

    return pipeline
