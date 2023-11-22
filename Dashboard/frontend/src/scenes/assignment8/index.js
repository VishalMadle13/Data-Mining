import React, { useEffect, useState } from 'react';

const Assignment8 = () => {
  const [defaultData, setDefaultData] = useState(null);

  useEffect(() => {
    // Fetch or set default data here
    const fetchData = async () => {
      try {
        // Simulating fetching data from an API
        const response = await fetch('http://127.0.0.1:8000/api/crawler/'); // Update with your API endpoint
        const data = await response.json();
        setDefaultData(data);
      } catch (error) {
        console.error('Error fetching default data:', error);
      }
    };

    fetchData();
  }, []);

  return (
    <div>
      {defaultData && (
        <div style={{ fontFamily: 'Arial, sans-serif' }}>
          <h1>Web Mining | Stanford Large Network Dataset</h1>
          <p>Root URL: http://snap.stanford.edu/data/#web</p>

          <h2>Page URLs</h2>
          <ul>
            {defaultData.page_urls.map((url, index) => (
              <li key={index}>{url}</li>
            ))}
          </ul>

          <h2>Hub Scores</h2>
          <p>{JSON.stringify(defaultData.hub_scores)}</p>

          <h2>Authority Scores</h2>
          <p>{JSON.stringify(defaultData.authority_scores)}</p>

          <h2>Top Pages by HITS</h2>
          <ul>
            {defaultData.top_pages_by_HITS.map((page, index) => (
              <li key={index}>{page}</li>
            ))}
          </ul>

          <h2>Page Rank by PR</h2>
          <p>{JSON.stringify(defaultData.page_rank_by_pr)}</p>

          <h2>Top Pages by PR</h2>
          <ul>
            {defaultData.top_pages_by_pr.map((page, index) => (
              <li key={index}>{page}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default Assignment8;
