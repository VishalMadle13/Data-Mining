import React from 'react';

function ClusterResult({ results }) {
  return (
    <div>
      <h2>Clustering Results</h2>
      {/* Display dendrograms and other clustering results */}
      <div>
        <h3>AGNES Dendrogram</h3>
        <img src={results.agnes_dendrogram} alt="AGNES Dendrogram" />
      </div>
      <div>
        <h3>DIANA Dendrogram</h3>
        <img src={results.diana_dendrogram} alt="DIANA Dendrogram" />
      </div>
      {/* Display k-Means, k-Medoids, BIRCH, and DBSCAN results here */}
      <div>
        <h3>k-Means Clusters</h3>
        {/* Display k-Means cluster assignments */}
      </div>
      <div>
        <h3>k-Medoids Clusters</h3>
        {/* Display k-Medoids cluster assignments */}
      </div>
      <div>
        <h3>BIRCH Clusters</h3>
        {/* Display BIRCH cluster assignments */}
      </div>
      <div>
        <h3>DBSCAN Clusters</h3>
        {/* Display DBSCAN cluster assignments */}
      </div>
      {/* Display cluster validation accuracy */}
      <div>
        <h3>Cluster Validation Accuracy</h3>
        {/* Display accuracy information */}
      </div>
    </div>
  );
}

export default ClusterResult;
