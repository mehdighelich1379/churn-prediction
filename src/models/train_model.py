import matplotlib.pyplot as plt
import joblib
import pandas as pd

def train_model(pipeline, x_train, y_train):
    pipeline.fit(x_train, y_train)

    # Extract transformed feature names
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get numeric and one-hot encoded feature names
    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]

    if hasattr(cat_encoder, 'get_feature_names_out'):
        cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
    else:
        # Older versions of sklearn
        cat_feature_names = cat_encoder.get_feature_names(cat_features)

    all_features = list(num_features) + list(cat_feature_names)

    # Get feature importances from final model
    model = pipeline.named_steps['model']
    feature_importances = pd.Series(model.feature_importances_, index=all_features)

    # Plot
    feature_importances.sort_values(ascending=True).plot(kind='barh', figsize=(12, 5), color='skyblue')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature')
    plt.title(f'Feature Importances - {type(model).__name__}')
    plt.tight_layout()
    plt.show()


    # Save the trained model
    save_path = 'src\models'
    joblib.dump(pipeline, f'{save_path}\{type(model).__name__}.joblib')

    return pipeline

