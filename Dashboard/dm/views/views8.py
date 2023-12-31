#----------------------------------------------------------------------------------------------------------------------------------#

from rest_framework.views import APIView
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from rest_framework import status

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import numpy as np
import json
import pickle
 
class WebCrawler:
    def __init__(self):
        self.graph = {}  # Adjacency list to store the graph
        self.visited = set()  # Set to keep track of visited pages

    def crawl(self, seed_url, max_depth=2):
        self._crawl(seed_url, depth=0, max_depth=max_depth)

    def _crawl(self, url, depth, max_depth):
        if depth > max_depth or url in self.visited:
            return

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract links from the page
            links = [urljoin(url, link.get('href')) for link in soup.find_all('a')]

            # Update the graph
            self.graph[url] = links

            # Mark the page as visited
            self.visited.add(url)

            # Recursively crawl linked pages
            for link in links:
                self._crawl(link, depth + 1, max_depth)

        except Exception as e:
            print(f"Error crawling {url}: {e}")
            
    def get_graph(self):
        return self.graph

class PageRankAlgorithm:
    @staticmethod
    def pagerank(adjacency_matrix, num_iterations=100, damping_factor=0.85):
        # Initialize the PageRank values
        num_pages = len(adjacency_matrix)
        page_rank = np.ones(num_pages) / num_pages

        for _ in range(num_iterations):
            new_page_rank = (1 - damping_factor) / num_pages + damping_factor * np.dot(adjacency_matrix.T, page_rank)
            if np.allclose(new_page_rank, page_rank, atol=1e-6):
                break
            page_rank = new_page_rank

        return page_rank

    @staticmethod
    def get_top_pages(page_rank, page_urls, top_n=10):
        # Get the indices of the top N pages
        top_indices = np.argsort(page_rank)[::-1][:top_n]

        # Get the page URLs and corresponding ranks
        top_pages = [(page_urls[i], page_rank[i]) for i in top_indices]

        return top_pages

class HITsAlgorithm:
    @staticmethod
    def hits(adjacency_matrix, num_iterations=100):
        num_pages = len(adjacency_matrix)

        # Initialize hub and authority scores
        hub_scores = np.ones(num_pages)
        authority_scores = np.ones(num_pages)

        for _ in range(num_iterations):
            # Update authority scores
            new_authority_scores = np.dot(adjacency_matrix.T, hub_scores)

            # Update hub scores
            new_hub_scores = np.dot(adjacency_matrix, new_authority_scores)

            # Normalize scores
            authority_scores = new_authority_scores / np.linalg.norm(new_authority_scores, 2)
            hub_scores = new_hub_scores / np.linalg.norm(new_hub_scores, 2)

        return hub_scores, authority_scores

    @staticmethod
    def get_top_pages(scores, page_urls, top_n=10):
        # Get the indices of the top N pages
        top_indices = np.argsort(scores)[::-1][:top_n]

        # Get the page URLs and corresponding scores
        top_pages = [(page_urls[i], scores[i]) for i in top_indices]

        return top_pages



class Crawler(APIView): 
    @method_decorator(csrf_exempt)
    def post(self, request, *args, **kwargs):
        if request.method == 'POST':
            try:
                root_url = request.POST.get("root-url")
                Crawler = WebCrawler()   
                Crawler.crawl(root_url)
                # Create a graph from the crawled data
                graph = Crawler.get_graph()

                # Transform the graph into an adjacency matrix
                num_pages = len(graph)
                page_urls = list(graph.keys())
                index_dict = {page: i for i, page in enumerate(page_urls)}

                adjacency_matrix = np.zeros((num_pages, num_pages))
                for i, links in enumerate(graph.values()):
                    for link in links:
                        if link in index_dict:
                            j = index_dict[link]
                            adjacency_matrix[i, j] = 1


                # Calculate PageRank using HITS algorithm
                hits_algorithm = HITsAlgorithm()
                hub_scores, authority_scores = hits_algorithm.hits(adjacency_matrix)
                top_pages_by_HITS = HITsAlgorithm.get_top_pages(authority_scores, page_urls)

                # Calculate PageRank using Pagerank algorithm
                pr_algorithm = PageRankAlgorithm()
                page_rank_by_pr = pr_algorithm.pagerank(adjacency_matrix.T)
                top_pages_by_pr = PageRankAlgorithm.get_top_pages(page_rank_by_pr, page_urls)

                result = {
                    "graph": graph,
                    "page_urls" : page_urls,
                    "hub_scores": hub_scores.tolist(),
                    "authority_scores": authority_scores.tolist(),
                    "top_pages_by_HITS": top_pages_by_HITS,
                    "page_rank_by_pr": page_rank_by_pr.tolist(),
                    "top_pages_by_pr": top_pages_by_pr
                } 

                return JsonResponse( result) 
            except Exception as e :
                return e
    def get(self, request, *args, **kwargs):
        if request.method =='GET':
            try:
                # open json file
                with open('results.json','r') as f:
                    data=json.load(f)
                    return JsonResponse(data)
            except Exception as e :
                return JsonResponse(e)

#----------------------------------------------------------------------------------------------------------------------------------#




 

 