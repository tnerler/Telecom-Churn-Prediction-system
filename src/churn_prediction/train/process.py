import pandas as pd 

from src.churn_prediction.features import feature_config


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def process_data(data: pd.DataFrame): 
    """
    To make ready data to make predictions or training;
    - Implement feature engineering
    - Select the columns that are necessery,
    - Turn categorical features that is datatype integer into categorical data type,
    - If there is missing data, fill/drop them

    Return processed data, as ready for the column transformer pipeline model.
    """

    # Creating new features
    cat_features = feature_config.categorical_features
    num_features = feature_config.numeric_features + feature_config.engineered_num_features
    

    # Building the column transformer pipeline

    categorical_transformer = Pipeline(
        steps= [
            (
                "onehot", 
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                num_features
            ),
            (
                "cat",
                categorical_transformer,
                cat_features
            )
        ]
    )

    return preprocessor

if __name__ == "__main__": 
    from src.churn_prediction.train.data_loader import load_data

    data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data = load_data(data_path)

    preprocessor = process_data(data)

    print(preprocessor)