from sklearn.datasets import load_breast_cancer
import pandas as pd
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()

# Save the data and target to CSV files
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['target'] = breast_cancer.target
data.to_csv('breast_cancer_dataset.csv', index=False)
