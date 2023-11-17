from rest_framework.views import APIView
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from rest_framework import status
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import base64
from sklearn.cluster import KMeans, DBSCAN
# from sklearn_extra.cluster import KMedoids
from sklearn.cluster import Birch
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES


class Clustering(APIView):
    def __init__(self):
        self.df = None

    @method_decorator(csrf_exempt)
    def post(self, request, *args, **kwargs):
        if request.method == 'POST':
            try:
                # Get the uploaded CSV file from the request
                file = request.FILES.get('file')
                # Read the file name of the uploaded file
                filename = file._name
                self.df = pd.read_csv(file)
                print(filename)

                if filename == "iris.csv":
                    # Load the Iris dataset
                    iris = load_iris()

                    # Create a DataFrame from the dataset
                    self.df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

                result = {
                    "agnes_dendrogram": self.AGNES_Dendrogram(),
                    "diana_dendrogram": self.DIANA_Dendrogram(),
                    "k_means_clusters": self.k_Means_Clustering(),
                    "k_medoids_clusters": self.k_Medoids_Clustering(),
                    "birch_clusters": self.BIRCH_Clustering(),
                    "dbscan_clusters": self.DBSCAN_Clustering(),
                }

                # Tabulate the results with cluster validation accuracy
                accuracy = self.calculate_cluster_accuracy()
                result["cluster_accuracy"] = accuracy

                # Return the result as a JSON response
                return JsonResponse(result)
            except Exception as e:
                # Handle any exceptions and return an error response
                print("ERROR:", str(e))
                return JsonResponse({"error": str(e), "msg": "Please preprocess the data!"}, status=status.HTTP_200_OK)

    def k_Means_Clustering(self):
        # Perform k-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(self.df)
        return kmeans.labels_

    # def k_Medoids_Clustering(self):
    #     # Perform k-Medoids (PAM) clustering
    #     kmedoids = KMedoids(n_clusters=3, random_state=0)
    #     kmedoids.fit(self.df)
    #     return kmedoids.labels_

    def BIRCH_Clustering(self):
        # Perform BIRCH clustering
        birch = Birch(n_clusters=3)
        birch.fit(self.df)
        return birch.labels_

    def DBSCAN_Clustering(self):
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(self.df)
        return dbscan.labels_

    def calculate_cluster_accuracy(self):
        # Assuming you have ground truth labels for iris dataset
        true_labels = load_iris().target

        # Accuracy for k-Means
        kmeans_accuracy = metrics.adjusted_rand_score(true_labels, self.k_Means_Clustering())

        # Accuracy for k-Medoids (PAM)
        kmedoids_accuracy = metrics.adjusted_rand_score(true_labels, self.k_Medoids_Clustering())

        # Accuracy for BIRCH
        birch_accuracy = metrics.adjusted_rand_score(true_labels, self.BIRCH_Clustering())

        # Accuracy for DBSCAN
        dbscan_accuracy = metrics.adjusted_rand_score(true_labels, self.DBSCAN_Clustering())

        return {
            "k_means_accuracy": kmeans_accuracy,
            "k_medoids_accuracy": kmedoids_accuracy,
            "birch_accuracy": birch_accuracy,
            "dbscan_accuracy": dbscan_accuracy,
        }
    
    def k_Medoids_Clustering(self):
        # Convert DataFrame to a list of data points
        data_points = self.df.values.tolist()

        # Perform k-Medoids (PAM) clustering
        initial_medoids = [0, 1, 2]  # Initial medoids
        kmedoids_instance = kmedoids(data_points, initial_medoids)
        kmedoids_instance.process()

        # Get cluster results
        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()

        # Visualize the clustering (optional)
        visualizer = cluster_visualizer()
        visualizer.append_clusters(clusters, data=data_points)
        visualizer.append_cluster(medoids, marker='*', markersize=10)
        visualizer.show()

        # Return cluster labels
        cluster_labels = [0] * len(data_points)
        for cluster_id, cluster in enumerate(clusters):
            for data_point_index in cluster:
                cluster_labels[data_point_index] = cluster_id

        return cluster_labels

