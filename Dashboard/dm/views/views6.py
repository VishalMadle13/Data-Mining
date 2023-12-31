from sklearn.datasets import load_iris,load_breast_cancer
from rest_framework.views import APIView
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import Birch, DBSCAN
from sklearn import metrics
from rest_framework import status
import matplotlib.pyplot as plt
import base64
import io
import numpy as np

class Clustering(APIView):
    @method_decorator(csrf_exempt)
    def post(self, request, *args, **kwargs):
        if request.method == 'POST':
            try:
                # Get the uploaded CSV file from the request
                file = request.FILES.get('file')
                algorithm = request.POST.get('algorithm')
                # print(algorithm)
                # read the file name of the uploaded file
                filename = file.name
                df = pd.read_csv(file)

                if(filename == 'iris.csv'):
                    # Load the Iris dataset
                    iris = load_iris() 
                    # Create a DataFrame from the dataset
                    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
                elif(filename == 'breastCancer.csv'):
                    breastCancer = load_breast_cancer()
                    df = pd.DataFrame(data=breastCancer.data, columns=breastCancer.feature_names)
                # print(df.info())

                # Perform hierarchical clustering using AGNES method
                agnes_linkage = linkage(df, method='average', metric='euclidean')
                agnes_dendrogram_path = self.plot_dendrogram(agnes_linkage, df.index, 'AGNES Dendrogram')

                # Perform hierarchical clustering using DIANA method
                diana_linkage = linkage(df, method='ward', metric='euclidean')
                diana_dendrogram_path = self.plot_dendrogram(diana_linkage, df.index, 'DIANA Dendrogram')

                # Perform clustering
                k_means = KMeans(n_clusters=3)  
                k_means_clusters = k_means.fit_predict(df)
 
                k_medoids = KMedoids(n_clusters=3)   
                k_medoids_clusters = k_medoids.fit_predict(df)

                birch = Birch(n_clusters=3)   
                birch_clusters = birch.fit_predict(df)
 
                dbscan = DBSCAN(eps=1.0, min_samples=5)   
                dbscan_clusters = dbscan.fit_predict(df)

                # Calculate cluster validation accuracy
                k_means_accuracy = metrics.silhouette_score(df, k_means_clusters)
                k_medoids_accuracy = metrics.silhouette_score(df, k_medoids_clusters)
                birch_accuracy = metrics.silhouette_score(df, birch_clusters) 
                
                # Filter out noise points (cluster -1)
                valid_labels = dbscan_clusters[dbscan_clusters != -1]
                valid_samples = df[dbscan_clusters != -1]

                # Check if there are valid clusters
                if len(np.unique(valid_labels)) > 1:
                    dbscan_accuracy = metrics.silhouette_score(valid_samples, valid_labels)
                else:
                    dbscan_accuracy = 0   
                    
                if(algorithm == 'agens'):
                    return JsonResponse({"agnes_dendrogram_path": agnes_dendrogram_path,})
                elif algorithm == 'diana':
                    return JsonResponse({"diana_dendrogram_path": diana_dendrogram_path,})
                elif algorithm == 'kmeans':
                    return JsonResponse({"k_means_accuracy": k_means_accuracy, "k_means_clusters": k_means_clusters.tolist()})
                
                elif algorithm == 'kmedoids':
                    return JsonResponse({"k_medoids_accuracy": k_means_accuracy, "k_medoids_clusters": k_means_clusters.tolist()})
                
                elif algorithm == 'birch':
                    return JsonResponse({"birch_clusters": birch_clusters.tolist(), "birch_accuracy": birch_accuracy})
                
                elif algorithm == 'dbscan':
                    return JsonResponse({"dbscan_clusters": dbscan_clusters.tolist(), "dbscan_accuracy": dbscan_accuracy})
                
            # -------------------- Plot and save scatter plot for clustering Algos ------------------------
                k_means_scatter_img = self.plot_scatter(df.iloc[:, 0], df.iloc[:, 1], k_means_clusters, 'K-Means Clustering')
                k_medoids_scatter_img = self.plot_scatter(df.iloc[:, 0], df.iloc[:, 1], k_medoids_clusters, 'K-Medoids Clustering')
                birch_scatter_img = self.plot_scatter(df.iloc[:, 0], df.iloc[:, 1], birch_clusters, 'BIRCH Clustering')
                dbscan_scatter_img = self.plot_scatter(df.iloc[:, 0], df.iloc[:, 1], dbscan_clusters, 'DBSCAN Clustering')
            # ---------------------------------------------------------------------------------------------

                result = {
                    "agnes_dendrogram_path": agnes_dendrogram_path,
                    "diana_dendrogram_path": diana_dendrogram_path,
                    "k_means_clusters": k_means_clusters.tolist(),
                    "k_means_scatter_img": k_means_scatter_img,
                    "k_medoids_clusters": k_medoids_clusters.tolist(),
                    "k_medoids_scatter_img": k_medoids_scatter_img,
                    "birch_clusters": birch_clusters.tolist(),
                    "birch_scatter_img": birch_scatter_img,
                    "dbscan_clusters": dbscan_clusters.tolist(),
                    "dbscan_scatter_img": dbscan_scatter_img,
                    "k_means_accuracy": k_means_accuracy,
                    "k_medoids_accuracy": k_medoids_accuracy,
                    "birch_accuracy": birch_accuracy,
                    "dbscan_accuracy": dbscan_accuracy,
                }

                return JsonResponse(result)
            except Exception as e:
                # Handle any exceptions and return an error response
                return JsonResponse({"error": str(e)}, status=status.HTTP_200_OK)
    def plot_scatter(self, x, y, clusters, title):
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.scatter(x, y, c=clusters, cmap='viridis', s=50)
        plt.xlabel('Feature X')
        plt.ylabel('Feature Y')
        plt.colorbar(label='Cluster')
        
        # Save the scatter plot image as a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # Convert the BytesIO object to a base64 encoded string
        img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close()

        return img_str
    def plot_dendrogram(self, linkage_matrix, labels, title):
        plt.figure(figsize=(12, 6))
        plt.title(title)
        dendrogram(linkage_matrix, orientation='top', labels=labels)
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        dendrogram_path = f'{title.lower().replace(" ", "_")}_dendrogram.png'
        plt.savefig(dendrogram_path)
        plt.close()

        # Convert the image to base64 encoded string
        img_str = None
        with open(dendrogram_path, "rb") as img_file:
            img_str = base64.b64encode(img_file.read()).decode('utf-8')

        return img_str

 