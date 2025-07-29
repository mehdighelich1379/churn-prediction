from sklearn.metrics import roc_curve, confusion_matrix, auc, classification_report
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def show_results(model , x_test , y_test , threshold=0.5):
    
    y_pred_probs = model.predict_proba(x_test)[:, 1]
    y_pred = (y_pred_probs > threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
    plt.legend(); plt.grid(); plt.show()

    

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

def Cv_results(model, X, Y, cv=5):
    scores = cross_validate(model, X, Y, cv=cv, scoring=['accuracy', 'f1', 'roc_auc', 'precision', 'recall'])
    df = pd.DataFrame(scores)
    return df
