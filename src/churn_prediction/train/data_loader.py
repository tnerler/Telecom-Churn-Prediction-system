import pandas as pd

def load_data(data_path: str): 

	"""
	Returns the data as pd.DataFrame.
	"""

	data = pd.read_csv(data_path)
	return data

if __name__ == "__main__": 
	data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
	data  = load_data(data_path)
    
