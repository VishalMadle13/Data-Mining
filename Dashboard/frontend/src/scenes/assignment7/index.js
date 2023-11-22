import React, { useState, useEffect } from 'react';
import AssociationRulesTable from './AssociationRulesTable'; // Adjust the path as needed

const Assignement7 = () => {
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch your results on component mount
    const fetchData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/run_association_rules/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          // Replace 'your_data' with the actual data you want to send in the request
          body: JSON.stringify({ support_values: [0.3, 0.45, 0.5],
          confidence_values: [0.5, 0.6, 0.7] }),
        });

        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }

        const data = await response.json();
        setResults(data);
      } catch (error) {
        console.error('Error fetching data:', error);
        setError('Failed to fetch data');
      }
    };

    fetchData();
  }, []);

  return (
    <div>
      {/* Other components or content */}
      {error && <p>Error: {error}</p>}
      <AssociationRulesTable results={results} />
    </div>
  );
};

export default Assignement7;
