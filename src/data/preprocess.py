import pandas as pd
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess(base_model , data_path , random_state=42 , test_size = 0.2):
    data = pd.read_csv(data_path)
    data.dropna(inplace=True)
    X = data.drop(columns=['CustomerID' , 'Churn'] , axis=1)
    Y = data.Churn

    num_col = [col for col in X.columns if X[col].dtype != 'O']
    cat_col = [col for col in X.columns if X[col].dtype == 'O']

    pre = ColumnTransformer(transformers=[('scale' , StandardScaler() , num_col) , ('le', OneHotEncoder(sparse=False, handle_unknown='ignore') , cat_col)])

    pipeline = Pipeline(steps=[('preprocessor' , pre) , ('model' , base_model)])


    x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 42)

    return pipeline , x_train , x_test , y_train , y_test , X ,Y



