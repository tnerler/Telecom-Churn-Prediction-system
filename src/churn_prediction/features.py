def create_features(df):

    df = df.copy()

    service_features = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    active_values = {"Yes", "DSL", "Fiber optic"}

    df["service_count"] = (
        df[service_features]
        .isin(active_values)
        .sum(axis=1)
    )

    df["services_per_month"] = (
        df["service_count"] /
        df["tenure"].replace(0, 1)
    )




    return (
        df["service_count"], 
        df["services_per_month"]
    ), () # First one is numerical, the second one is categorical features



class FeatureConfig: 

    categorical_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"

    ]

    engineered_num_features = [
        "service_count",
        "services_per_month"
    ]

    target_variable = [
        "Churn"
    ]

    dropped_features = ["customerID"]

feature_config = FeatureConfig()


