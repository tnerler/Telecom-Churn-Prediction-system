from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np 


def split_the_data(data: pd.DataFrame, 
                   config): 
    """
    1- Split the data into features and target.
    2- Split the data into training and testing data.
    """
    np.random.seed(config["split"]["random_state"])

    X = data.drop(["Churn"], axis=1)
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=config["split"]["test_size"],
                                                        stratify=y 
                                                        )
    
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__": 
    import yaml 
    from src.churn_prediction.train.data_loader import load_data

    data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_data(data_path)


    # Open the file in read mode
    with open("configs/config.yaml", "r") as stream:
        try:
            # Load and parse the YAML file
            data = yaml.safe_load(stream)
            # print(data)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")

    train, test = split_the_data(df, data)

    
    