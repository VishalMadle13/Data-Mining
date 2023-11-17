# views.py
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
import plotly.graph_objs as go
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import JsonResponse
from django.views import View
import statistics



class CSVUploadView(APIView):
    parser_classes = (FileUploadParser,)

    @method_decorator(csrf_exempt)
    def post(self, request, *args, **kwargs):
        file = request.FILES.get('file')

   
        if file is None:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        response_data = self.calculate_statistics(file)
        return JsonResponse(response_data)

    def calculate_statistics(self, file):
        data = []
        columns = {}

        # Read CSV file and store data
        for line in file:
            line = line.decode('utf-8')
            row = line.strip().split(',')
            data.append(row)
        # Get column names from the first row
        col_names = [i for i in range(len(data[4])-1)]

        # Initialize empty lists for each column
        for col_name in col_names:
            columns[col_name] = []

        # Populate column data
        for row in data[1:]:
            for i, val in enumerate(row):
                try:
                    if i < len(col_names) :
                        columns[i].append(float(val))
                except Exception as e:
                    pass

        # Calculate statistics for each column
        stats = {}
        for col_name, col_data in columns.items():

            try :
                mean = sum(col_data) / len(col_data)
                median = self.calculate_median(col_data)
                mode = self.calculate_mode(col_data)
                midrange = (min(col_data) + max(col_data)) / 2
                variance = self.calculate_variance(col_data, mean)
                std_deviation = variance ** 0.5

                stats[col_name] = {
                    'mean': format(mean, ".2f"),
                    'median': format(median, ".2f"),
                    'mode': format(mode, ".2f"),
                    'midrange': format(midrange, ".2f"),
                    'variance': format(variance, ".2f"),
                    'std_deviation': format(std_deviation, ".2f"),
                    'range': format(self.range(col_data), ".2f"),
                    'quartiles': self.quartiles(col_data),
                    'IQR': format(self.interquartile_range(col_data), ".2f"),
                    'five_number_summary': self.five_number_summary(col_data),
                }
            except Exception as e :
                pass

        return {
            'statistics': stats
            }

    def calculate_median(self, data):
        # Implementation for calculating median
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 0:
            middle_indices = [n // 2 - 1, n // 2]
            median = sum(sorted_data[i] for i in middle_indices) / 2
        else:
            median = sorted_data[n // 2]
        return median

        # Implementation for calculating mode
        # ...

    def calculate_variance(self, data, mean):
        # Implementation for calculating variance
        squared_deviations = [(value - mean) ** 2 for value in data]
        variance = sum(squared_deviations) / len(data)
        
        return variance



    def calculate_mode(self, data):
        frequency = {}
        for value in data:
            if value in frequency:
                frequency[value] += 1
            else:
                frequency[value] = 1

        mode = max(frequency, key=frequency.get)
        return mode

    def range(self, data):
        return max(data) - min(data) if data else 0

    def quartiles(self, data):
        data.sort()
        n = len(data)
        q1_index = n // 4
        q2_index = n // 2
        q3_index = 3 * n // 4
        return data[q1_index], data[q2_index], data[q3_index]

    def interquartile_range(self, data):
        q1, _, q3 = self.quartiles(data)
        return q3 - q1

    def five_number_summary(self, data):
        data.sort()
        n = len(data)
        q1, q2, q3 = self.quartiles(data)
        return data[0], q1, q2, q3, data[-1]

