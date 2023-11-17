 
# Load CSV data
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris

df = pd.read_csv('D:\\WCE\\BTECH SEM 7\\DM\\ASSIGNMENT\\2020BTECS00092_LA1\\Assignments\\Dashboard\\Dashboard\\datasets\\iris.csv')
 
 
# Use LabelEncoder to convert the "variety" column to numeric
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['variety'] = label_encoder.fit_transform(df['variety'])

# Display the first few rows of the DataFrame
print(df.head())
