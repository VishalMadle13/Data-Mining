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

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_iris
import graphviz


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
import matplotlib
matplotlib.use('Qt5Agg')
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

class InfoGain(APIView):
    @method_decorator(csrf_exempt)
    def post(self, request, *args, **kwargs):
        if request.method == 'POST':
            try:
                file = request.FILES.get('file')
                df = pd.read_csv(file)

                df = df.dropna(axis=0)

                target_class = df.columns[-1]

                X = df.drop(columns=[target_class])
                y = df[target_class]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                results = {}

                # print(X.head())
                # print(y.head())

                print("test")

                clf_info_gain = DecisionTreeClassifier(criterion="entropy")
                clf_info_gain.fit(X_train, y_train)

                y_pred = clf_info_gain.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)

                coverage = np.sum(np.diag(cm)) / np.sum(cm)
                accuracy = accuracy_score(y_test, y_pred)
                toughness = clf_info_gain.tree_.node_count

                results['entropy'] = {
                    'confusion_matrix': cm.tolist(),
                    'coverage': coverage,
                    'toughness': toughness
                }

               
                clf_gain_ratio = DecisionTreeClassifier(criterion="entropy", splitter="best")
                clf_gain_ratio.fit(X_train, y_train)
                y_pred = clf_gain_ratio.predict(X_test)
                coverage = np.sum(np.diag(cm)) / np.sum(cm)
                accuracy = accuracy_score(y_test, y_pred)
                toughness = clf_gain_ratio.tree_.node_count

                results['gain'] = {
                    'confusion_matrix': cm.tolist(),
                    'coverage': coverage,
                    'toughness': toughness
                }

                clf_gini = DecisionTreeClassifier(criterion="gini")
                clf_gini.fit(X_train, y_train)
                y_pred = clf_gini.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                coverage = np.sum(np.diag(cm)) / np.sum(cm)
                accuracy = accuracy_score(y_test, y_pred)
                toughness = clf_gini.tree_.node_count

                results['gini'] = {
                    'confusion_matrix': cm.tolist(),
                    'coverage': coverage,
                    'toughness': toughness
                }


                # Visualize the Information Gain decision tree
                plt.figure(figsize=self.calculate_figsize(clf_info_gain))
                plot_tree(clf_info_gain, filled=True, feature_names=df.columns[:-1], class_names=df[target_class].unique(), rounded=True, fontsize=10)
                plt.title("Information Gain")
                plt.savefig("C:\\Academics\\B. Tech\DM Lab\\Assignment 1\\Dashboard\\frontend\\src\\decision_tree1.png")  # Save as an image
                plt.close(plt.gcf())
                # plt.close() 

                # Visualize the Gain Ratio decision tree
                plt.figure(figsize=self.calculate_figsize(clf_gain_ratio))
                plot_tree(clf_gain_ratio, filled=True, feature_names=df.columns[:-1], class_names=df[target_class].unique(), rounded=True, fontsize=10)
                plt.title("Gain Ratio")
                plt.savefig("C:\\Academics\\B. Tech\DM Lab\\Assignment 1\\Dashboard\\frontend\\src\\decision_tree2.png")  # Save as an image
                plt.close(plt.gcf())
                # plt.close()

                # Visualize the Gini Index decision tree
                plt.figure(figsize=self.calculate_figsize(clf_gini))
                plot_tree(clf_gini, filled=True, feature_names=df.columns[:-1], class_names=df[target_class].unique(), rounded=True, fontsize=10)
                plt.title("Gini Index")
                plt.savefig("C:\\Academics\\B. Tech\DM Lab\\Assignment 1\\Dashboard\\frontend\\src\\decision_tree3.png")  # Save as an image
                plt.close(plt.gcf())
                # plt.close()
                



                # dot_info_gain = export_graphviz(clf_info_gain, out_file=None, feature_names=df.columns[:-1], class_names=df[target_class].unique(), filled=True, rounded=True)
                # dot_gain_ratio = export_graphviz(clf_gain_ratio, out_file=None, feature_names=df.columns[:-1], class_names=df[target_class].unique(), filled=True, rounded=True)
                # dot_gini = export_graphviz(clf_gini, out_file=None, feature_names=df.columns[:-1], class_names=df[target_class].unique(), filled=True, rounded=True)
                # print("test")

                # graph_info_gain = graphviz.Source(dot_info_gain)
                # graph_gain_ratio = graphviz.Source(dot_gain_ratio)
                # graph_gini = graphviz.Source(dot_gini)

                # print("test")
                # # Save or render the tree visualization
                # graph_info_gain.save("decision_tree_info_gain")
                # graph_gain_ratio.save("decision_tree_gain_ratio")
                # graph_gini.save("decision_tree_gini")
                                
                

                return JsonResponse(results,safe=False)
            except Exception as e :
                return JsonResponse({"error => ": str(e)}, status=status.HTTP_200_OK, safe=False)
    
    def calculate_figsize(self, tree):
        # Determine the depth of the tree
        depth = tree.tree_.max_depth

        # Calculate the number of nodes (leaves + internal nodes)
        num_nodes = 2 ** (depth + 1) - 1

        # Determine an appropriate figsize based on the number of nodes
        # You can adjust these constants based on your preferences
        figsize = (num_nodes * 0.1, depth * 2)

        return figsize