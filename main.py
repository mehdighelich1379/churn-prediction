import pandas as pd
import joblib
from xgboost import XGBClassifier
from src.data.preprocess import preprocess
from src.models.train_model import train_model
from src.utils.metrics import show_results, Cv_results

def main():
    data_path = "data/customer_churn_dataset-training-master.csv"
    base_model= XGBClassifier(
        n_estimators=300,        # تعداد درخت‌ها
        learning_rate=0.05,       # نرخ یادگیری
        max_depth=6,             # بیشینه عمق هر درخت
        subsample=0.8,           # درصد نمونه‌گیری از داده‌ها برای هر درخت
        gamma=0,                 # حداقل کاهش لازم برای تقسیم گره
        reg_alpha=0,             # L1 regularization term
        reg_lambda=1,            # L2 regularization term
        use_label_encoder=False, # جلوگیری از اخطار نسخه‌های قدیمی
        eval_metric='logloss',   # معیار ارزیابی
        random_state=42          # بازتولیدپذیری
    )

    print("🔧 Training on old data...")
    pipeline , x_train , x_test , y_train , y_test  , X , Y = preprocess(base_model , data_path , random_state=42 , test_size = 0.3)
    pipeline = train_model(pipeline, x_train, y_train)

    show_results(pipeline , x_test , y_test , threshold=0.5)

    print("\n📈 Cross Validation:")
    df = Cv_results(model=pipeline , X=X , Y=Y )
    df.to_csv('src//utils/Cv_results.csv')


if __name__ == "__main__":
    main()
