 
// import React, { useState } from 'react';
// import axios from 'axios';

// import './Assignment6.css'; // Import your CSS file for styling

// function Assignment6() {
// const [csvFile, setCsvFile] = useState(null);
// const [agnesDendrogram, setAgnesDendrogram] = useState(null);
// const [dianaDendrogram, setDianaDendrogram] = useState(null);
// const [kMeansResult, setKMeansResult] = useState(null);
// const [kMedoidsResult, setKMedoidsResult] = useState(null);
// const [birchResult, setBirchResult] = useState(null);

// const handleFileUpload = (e) => {
//   const file = e.target.files[0];
//   setCsvFile(file);
// };

// const handleUpload = () => {
//   const formData = new FormData();
//   formData.append('file', csvFile);

//   axios.post('http://http://127.0.0.1:8000/api/clustering/', formData)
//     .then((response) => {
//       const data = response.data;
//       setAgnesDendrogram(data.agnes_dendrogram);
//       setDianaDendrogram(data.diana_dendrogram);
//       setKMeansResult(data.kMeansResult); // Replace with actual data for k-Means
//       setKMedoidsResult(data.kMedoidsResult); // Replace with actual data for k-Medoids
//       setBirchResult(data.birchResult); // Replace with actual data for BIRCH
//     })
//     .catch((error) => {
//       console.error('Error uploading CSV and fetching results', error);
//     });
// };

// return (
//   <div className="assignment6-container">
//     <h2>CSV Data Upload</h2>
//     <label htmlFor="file-upload" className="custom-file-upload">
//       <input type="file" id="file-upload" accept=".csv" onChange={handleFileUpload} />
//       Choose a CSV File
//     </label>

//     <div>
//       {agnesDendrogram && (
//         <div className="result-card">
//           <h3>AGNES Dendrogram</h3>
//           <img src={agnesDendrogram} alt="AGNES Dendrogram" />
//         </div>
//       )}

//       {dianaDendrogram && (
//         <div className="result-card">
//           <h3>DIANA Dendrogram</h3>
//           <img src={dianaDendrogram} alt="DIANA Dendrogram" />
//         </div>
//       )}

//       {kMeansResult && (
//         <div className="result-card">
//           <h3>k-Means Results</h3>
//           {/* Display k-Means clustering results here */}
//         </div>
//       )}

//       {kMedoidsResult && (
//         <div className="result-card">
//           <h3>k-Medoids Results</h3>
//           {/* Display k-Medoids clustering results here */}
//         </div>
//       )}

//       {birchResult && (
//         <div className="result-card">
//           <h3>BIRCH Results</h3>
//           {/* Display BIRCH clustering results here */}
//         </div>
//       )}

//       {/* Add buttons for other clustering types and display cluster validation results accordingly */}
//       <div className="button-container">
//         <button onClick={handleUpload}>Upload CSV and Run Clustering</button>
//         {/* <button onClick={handleKMeans}>Run k-Means Clustering</button>
//         <button onClick={handleKMedoids}>Run k-Medoids Clustering</button>
//         <button onClick={handleBIRCH}>Run BIRCH Clustering</button> */}
//       </div>
//     </div>
//   </div>
// );
// }

// export default Assignment6;
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Papa from 'papaparse';
import ClusteringResult from './ClusteringResult';

function Assignment6() {
  const [csvData, setCsvData] = useState([]);
  const [csvFile, setCsvFile] = useState(null);
  const [results, setResults] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/api/clustering/');
      const csvData = parseCsvData(response.data);
      setCsvData(csvData);
    } catch (error) {
      console.error('Error fetching .csv data', error);
    }
  };

  const parseCsvData = (csvText) => {
    const parsed = Papa.parse(csvText, { header: true });
    return parsed.data;
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const formData = new FormData();
      formData.append('file', file);

      axios.post('http://127.0.0.1:8000/api/clustering/', formData).then((response) => {
        setResults(response.data);
      });
    }
  };

  return (
    <>
      <div>
        <h2>CSV Data Display</h2>
        <input type="file" accept=".csv" onChange={handleFileChange} />
        <ClusteringResult results={results} />
      </div>
    </>
  );
}

export default Assignment6;
