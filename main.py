import pandas as pd
import joblib
from xgboost import XGBClassifier
from src.data.preprocess import preprocess
from src.models.train_model import train_model
from src.utils.metrics import show_results, Cv_results
from lightgbm import LGBMClassifier

def main():
    data_path = "data/customer_churn_dataset-training-master.csv"
    base_model= XGBClassifier(
        n_estimators=300,        
        learning_rate=0.05,       
        max_depth=6,             
        subsample=0.8,            
        gamma=0,                 
        reg_alpha=0,             # L1 regularization term
        reg_lambda=1,            # L2 regularization term
        use_label_encoder=False, 
        eval_metric='logloss',  
        random_state=42         
    )
#     base_model = LGBMClassifier(
#     n_estimators=100,        
#     learning_rate=0.1,     
#     max_depth=-1,            
#     num_leaves=31,            
#     subsample=0.8,            
#     colsample_bytree=0.8,    
#     reg_alpha=0.0,           
#     reg_lambda=1.0,      
#     min_child_samples=20,
#     objective='binary',      
#     boosting_type='gbdt',  
#     random_state=42     
# )

    print("🔧 Training on old data...")
    pipeline , x_train , x_test , y_train , y_test  , X , Y = preprocess(base_model , data_path , random_state=42 , test_size = 0.3)
    pipeline = train_model(pipeline, x_train, y_train, x_test, y_test, experiment_name="Customer_Churn_Experiment")

    show_results(pipeline , x_test , y_test , threshold=0.5)

    print("\n📈 Cross Validation:")
    df = Cv_results(model=pipeline , X=X , Y=Y )
    df.to_csv('src//utils/Cv_results.csv')


if __name__ == "__main__":
    main()
