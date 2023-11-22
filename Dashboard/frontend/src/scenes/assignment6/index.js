import React, { useState } from 'react';
import axios from 'axios';
import { Box, Typography, Grid, Paper } from '@mui/material';
import { MuiFileInput } from 'mui-file-input';
import Plot from 'react-plotly.js';

const Assignment6 = () => {
  const [file, setFile] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [kMeansClusters, setKMeansClusters] = useState([]);
  const [birchClusters, setBirchClusters] = useState([]);
  const [dbscanClusters, setDbscanClusters] = useState([]);
  const [kMedoidsClusters, setKMedoidsClusters] = useState([]);

  const [kMeansClustersImg, setKMeansClustersImg] = useState(null);
  const [birchClustersImg, setBirchClustersImg] = useState(null);
  const [dbscanClustersImg, setDbscanClustersImg] = useState(null);
  const [kMedoidsClustersImg, setKMedoidsClustersImg] = useState(null);

  const [agnesDendrogramPath, setAgnesDendrogramPath] = useState(null);
  const [dianaDendrogramPath, setDianaDendrogramPath] = useState(null);
  const [kMeansAccuracy, setKMeansAccuracy] = useState(null);
  const [birchAccuracy, setBirchAccuracy] = useState(null);
  const [dbscanAccuracy, setDbscanAccuracy] = useState(null);
  const [kMedoidsAccuracy, setKMedoidsAccuracy] = useState(null);

  const handleFileChange = async (e) => {
    if (e) {
      setFile(e);
      setDataset(e.name);

      const formData = new FormData();
      formData.append('file', e);

      try {
        const response = await axios.post('http://127.0.0.1:8000/api/clustering/', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        setKMeansClusters(response.data.k_means_clusters);
        setBirchClusters(response.data.birch_clusters);
        setDbscanClusters(response.data.dbscan_clusters);
        setKMedoidsClusters(response.data.k_medoids_clusters);

        setKMeansClustersImg(response.data.k_means_scatter_img);
        setBirchClustersImg(response.data.birch_scatter_img);
        setDbscanClustersImg(response.data.dbscan_scatter_img);
        setKMedoidsClustersImg(response.data.k_medoids_scatter_img);

        setKMeansAccuracy(response.data.k_means_accuracy);
        setBirchAccuracy(response.data.birch_accuracy);
        setDbscanAccuracy(response.data.dbscan_accuracy);
        setKMedoidsAccuracy(response.data.k_medoids_accuracy);

        setAgnesDendrogramPath(response.data.agnes_dendrogram_path);
        setDianaDendrogramPath(response.data.diana_dendrogram_path);
      } catch (error) {
        console.error('Error performing clustering:', error);
      }
    }
  };

  const generateScatterPlot = (clusters, title, img) => (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        <Typography variant="h6">{title}</Typography>
        <Plot data={[{ x: Array.from({ length: clusters.length }, (_, i) => i), y: clusters, mode: 'markers', type: 'scatter', marker: { color: clusters, size: 10, colorscale: 'Viridis' } }]} />
      </Grid>
      <Grid item xs={12} md={6}>
        <img src={`data:image/png;base64,${img}`} alt={`${title} Scatter Plot`} style={{ width: '100%', borderRadius: '8px' }} />
      </Grid>
    </Grid>
  );

  return (
    <Box m={4}>
      <Typography variant="h4">Upload CSV File for Clustering</Typography>
      <MuiFileInput label="Upload File" value={file ? file.name : ''} type="file" onChange={handleFileChange} />

      <Typography variant="h4" mt={4}>Clustering Visualization || {dataset} dataset</Typography>

      <Box mt={4}>
        <Typography variant="h5">Hierarchical Clustering - AGNES</Typography>
        {agnesDendrogramPath && (
          <img src={`data:image/png;base64,${agnesDendrogramPath}`} alt="Agnes Dendrogram" style={{ width: '100%', borderRadius: '8px' }} />
        )}
      </Box>

      <Box mt={4}>
        <Typography variant="h5">Hierarchical Clustering - DIANA</Typography>
        {dianaDendrogramPath && (
          <img src={`data:image/png;base64,${dianaDendrogramPath}`} alt="Diana Dendrogram" style={{ width: '100%', borderRadius: '8px' }} />
        )}
      </Box>

      <Box mt={4}>
        <Typography variant="h5">k-Means</Typography>
        {kMeansClusters && generateScatterPlot(kMeansClusters, 'K-Means Clusters', kMeansClustersImg)}
        <Typography variant="h6"><span style={{ fontWeight: 'bold' }}>Kmean Accuracy: {kMeansAccuracy}</span></Typography>
      </Box>

      <Box mt={4}>
        <Typography variant="h5">k-Medoids (PAM)</Typography>
        {kMedoidsClusters && generateScatterPlot(kMedoidsClusters, 'K-Medoids Clusters', kMedoidsClustersImg)}
        <Typography variant="h6"><span style={{ fontWeight: 'bold' }}>k-Medoids Accuracy: {kMedoidsAccuracy}</span></Typography>
      </Box>

      <Box mt={4}>
        <Typography variant="h5">BIRCH</Typography>
        {birchClusters && generateScatterPlot(birchClusters, 'Birch Clusters', birchClustersImg)}
        <Typography variant="h6"><span style={{ fontWeight: 'bold' }}>Birch Accuracy: {birchAccuracy}</span></Typography>
      </Box>

      <Box mt={4}>
        <Typography variant="h5">DBSCAN</Typography>
        {dbscanClusters && generateScatterPlot(dbscanClusters, 'DB-Scan Clusters', dbscanClustersImg)}
        <Typography variant="h6"> <span style={{ fontWeight: 'bold' }}>DBSCAN Accuracy: {dbscanAccuracy} </span></Typography>
      </Box>
    </Box>
  );
};

export default Assignment6;
