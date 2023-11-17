# views.py

# Chi-Square Value: 1922.9347363945576

# P-Value: 2.6830523867648017e-17

from scipy.stats import chi2_contingency,zscore,pearsonr
import tempfile
from django.shortcuts import render
import json
# Create your views here.
from rest_framework.parsers import FileUploadParser
import csv
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.views import View
import csv
import math
from django.http import JsonResponse
from django.views import View
import csv
from django.http import HttpResponse
import json
from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import csv
import statistics
import numpy as np
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import JsonResponse
from django.views import View
import statistics
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency,zscore,pearsonr
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import datasets
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.tree import export_text
from django.views.decorators.csrf import csrf_exempt
import logging
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import tempfile
import shutil
import math
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView
from rest_framework import status
from django.http import JsonResponse, FileResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

import pandas as pd
import os

import numpy as np
from django.http import JsonResponse
from scipy.stats import chi2_contingency
from scipy.stats import chi2

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import graphviz
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle


class Regression(APIView):
    @method_decorator(csrf_exempt)
    def post(self, request, *args, **kwargs):
        if request.method == 'POST':
            try:
                
                file = request.FILES.get('file')
                algo = request.POST['algo']
                df = pd.read_csv(file)

                df = shuffle(df, random_state=42)

                target_class = df.columns[-1]

                object_cols = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype=='object' and col != target_class]
                numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != target_class]

                X = df[numeric_cols+object_cols]
                y = df[target_class]

                # print(X.head())
                # print(y.head())
                

                X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
                ordinal_encoder = OrdinalEncoder()

                if target_class in object_cols :
                    object_cols = [col for col in object_cols if col != target_class]
                    y_train[target_class] = OrdinalEncoder.fit_transform(y_train[target_class])
                    y_test[target_class] = OrdinalEncoder.fit_transform(y_test[target_class])

                X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
                X_test[object_cols] = ordinal_encoder.transform(X_test[object_cols])


                if algo == "Linear" : 
                    cm = self.logistic_regression(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                elif algo == "Naive" : 
                    cm = self.naive_classifier(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                elif algo == "KNN" : 
                    cm = self.knn_classifier(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                elif algo == "ANN" : 
                    cm = self.ann_classifier(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                
                return JsonResponse({"accuracy": "accuracy"})
            except Exception as e :
                print(e)
                return JsonResponse({"error => ": str(e)}, status=status.HTTP_200_OK, safe=False)

    def preprocess(self, df):
        numerical_columns = df.select_dtypes(include=[int, float])

        # Select columns with a number of unique values less than 4
        unique_threshold = 4
        selected_columns = []
        for column in df.columns:
            if len(df[column].unique()) < unique_threshold and df[column].dtype == 'object':
                selected_columns.append(column)

        # Combine the two sets of selected columns (numerical and unique value threshold)
        final_selected_columns = list(set(numerical_columns.columns).union(selected_columns))

        # Create a new DataFrame with only the selected columns
        filtered_df = df[final_selected_columns]

        from sklearn.preprocessing import LabelEncoder

        # Assuming 'filtered_df' is your DataFrame with object-type columns to be encoded
        encoder = LabelEncoder()

        for column in filtered_df.columns:
            if filtered_df[column].dtype == 'object':
                filtered_df[column] = encoder.fit_transform(filtered_df[column])
        
        return filtered_df


    def logistic_regression(self, X, y, X_train, X_test, y_train, y_test):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        # Load your dataset (e.g., Iris or Breast Cancer)
        # X, y, X_train, X_test, y_train, y_test = load_data()
        # Split the data into training and testing sets
        
        # Create and train the regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)

        print(y_pred.shape)
        print(y_test.shape)
        cm = confusion_matrix(y_test, y_pred).tolist()
        print(cm)

        
        return cm

    def naive_classifier(self, X, y, X_train, X_test, y_train, y_test):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score

        

        # Create and train the Naïve Bayes classifier
        nb_classifier = GaussianNB()
        nb_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = nb_classifier.predict(X_test)

        # Calculate accuracy
        cm = confusion_matrix(y_test, y_pred).tolist()

        return cm

    def knn_classifier(self, X, y, X_train, X_test, y_train, y_test):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score

        

        # Create and train the k-NN classifier with different values of k
        k_values = [1, 3, 5, 7]
        cm = []
        accuracy_scores = []


        for k in k_values:
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(X_train, y_train)
            
            # Make predictions
            y_pred = knn_classifier.predict(X_test)
            
            # Calculate accuracy
            metrix = confusion_matrix(y_test, y_pred).tolist()
            cm.append({'confusion_matrix': metrix})


            print(metrix)

            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # Plot the error graph
        plt.figure(figsize=(8, 6))  # Adjust figure size as needed
        plt.plot(k_values, accuracy_scores)
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        # plt.show()

        plt.savefig("D:\\WCE\\BTECH SEM 7\\s\\Dashboard\\Dashboard\\frontend\\src\\KNNplot.png")
    
        return cm

    def ann_classifier(self, X, y, X_train, X_test, y_train, y_test):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import load_iris, load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier

        

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the ANN classifier
        mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        mlp.fit(X_train, y_train)

        # Plot the error graph (iteration vs error)
        plt.figure(figsize=(8, 6))
        plt.plot(mlp.loss_curve_)
        plt.title('Error Graph (Iteration vs Error)')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)
        # plt.show()

        plt.savefig("D:\\WCE\\BTECH SEM 7\\s\\Dashboard\\Dashboard\\frontend\\src\\ANNplot.png")

        # Evaluate the classifier
        y_pred = mlp.predict(X_test)

        cm = confusion_matrix(y_test, y_pred).tolist()

        return cm
