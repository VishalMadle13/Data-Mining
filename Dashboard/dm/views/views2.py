# views.py

# Chi-Square Value: 1922.9347363945576

# P-Value: 2.6830523867648017e-17


import math
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView
from rest_framework import status
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

import pandas as pd
import numpy as np
from django.http import JsonResponse
from scipy.stats import chi2_contingency
from scipy.stats import chi2

class Chi_Analyze(APIView):
    @method_decorator(csrf_exempt)
    def post(self, request, *args, **kwargs):
        if request.method == 'POST':
            try:
                file = request.FILES.get('file')
                df = pd.read_csv(file)

                attribute1 = request.POST['attribute1']
                attribute2 = request.POST['attribute2']  

                contingency_table = self.create_contingency_table(df, attribute1, attribute2)

                print(type(contingency_table))
            

                chi_square, p, dof, expected = self.calculate_chi_square(contingency_table)

                result = {
                    'contingency_table': contingency_table,
                    'chi2_value': chi_square,
                    'p_value': p,
                    'correlation_result': 'Correlated' if p < 0.05 else 'Not Correlated'
                }

                return JsonResponse(result)
            except Exception as e :
                return JsonResponse({"error => ": str(e)}, status=status.HTTP_200_OK)

    def create_contingency_table(self, data, attribute1, attribute2):
        unique_values_attr1 = set(data[attribute1])
        unique_values_attr2 = set(data[attribute2])
        
        contingency_table = {}
        
        for value_attr1 in unique_values_attr1:
            contingency_table[value_attr1] = {}
            for value_attr2 in unique_values_attr2:
                contingency_table[value_attr1][value_attr2] = 0
        
        for index, row in data.iterrows():
            contingency_table[row[attribute1]][row[attribute2]] += 1
        
        return contingency_table

    def calculate_chi_square(self, contingency_table):
        rows = len(contingency_table)
        cols = len(contingency_table[next(iter(contingency_table))])

        # Calculate total observed frequency and row/column sums
        total_observed = 0
        row_sums = [0] * rows
        col_sums = [0] * cols

        for i, row_key in enumerate(contingency_table):
            for j, col_key in enumerate(contingency_table[row_key]):
                frequency = contingency_table[row_key][col_key]
                total_observed += frequency
                row_sums[i] += frequency
                col_sums[j] += frequency

        # Calculate expected values
        expected_values = []
        for i in range(rows):
            row_expected = []
            for j in range(cols):
                expected = (row_sums[i] * col_sums[j]) / total_observed
                row_expected.append(expected)
            expected_values.append(row_expected)

        # Calculate chi-square statistic
        chi_square = 0
        for i, row_key in enumerate(contingency_table):
            for j, col_key in enumerate(contingency_table[row_key]):
                observed = contingency_table[row_key][col_key]
                expected = expected_values[i][j]
                chi_square += ((observed - expected) ** 2) / expected

        # Calculate degrees of freedom
        dof = (rows - 1) * (cols - 1)

        # Calculate p-value (using chi-squared distribution)
        p_value = 1 - chi2.cdf(chi_square, dof)
        
        return chi_square, p_value, dof, expected_values





class Corelation_Analyze(APIView):
    @method_decorator(csrf_exempt)
    def post(self, request, *args, **kwargs):
        if request.method == 'POST':
            file = request.FILES['file']
            df = pd.read_csv(file)

            attribute1 = request.POST['attribute1']
            attribute2 = request.POST['attribute2']


            correlation_coefficient = self.calculate_correlation_coefficient(df[attribute1],df[attribute2])
            covariance = self.calculate_covariance(df[attribute1], df[attribute2])

            result = {
                'correlation_coefficient': correlation_coefficient,
                'covariance': covariance,
                'conclusion': 'Correlated' if abs(correlation_coefficient) > 0.5 else 'Not Correlated'
            }

            return JsonResponse(result)


    def calculate_mean(self, data):
        return sum(data) / len(data)

    def calculate_correlation_coefficient(self, x, y):
        mean_x = self.calculate_mean(x)
        mean_y = self.calculate_mean(y)
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator_x = sum((xi - mean_x)**2 for xi in x)
        denominator_y = sum((yi - mean_y)**2 for yi in y)
        
        correlation_coefficient = numerator / (denominator_x**0.5 * denominator_y**0.5)
        return correlation_coefficient

    def calculate_covariance(self, x, y):
        mean_x = self.calculate_mean(x)
        mean_y = self.calculate_mean(y)
        
        covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / len(x)
        return covariance



class PerformNormalization(APIView):
    @method_decorator(csrf_exempt)
    def post(self, request, *args, **kwargs):
        if request.method == 'POST':
            try:
                file = request.FILES['file']
                df = pd.read_csv(file)
                target_class = df.columns[-1]
                df = df.drop([target_class], axis=1)
                
                normalization_type = request.POST['normalization_type']

                df_numeric = df.apply(pd.to_numeric, errors='coerce')  # Convert data to numeric format

                if normalization_type == 'min_max':
                    df_normalized = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())
                elif normalization_type == 'z_score':
                    df_normalized = (df_numeric - df_numeric.mean()) / df_numeric.std()
                elif normalization_type == 'decimal_scaling':
                    scale_factor = 10 ** (int(request.POST['decimal_scale']))
                    df_normalized = df_numeric / scale_factor
                else:
                    return JsonResponse({'error': 'Invalid normalization type'})

                return JsonResponse({'normalized_data': df_normalized.to_dict(orient='list')})
            except Exception as e :
                return JsonResponse({'normalized_data': str(e)})
