import React, { useEffect, useState } from 'react';

const AssociationRulesTable = ({ results }) => {
  const [formattedData, setFormattedData] = useState([]);

  useEffect(() => {
    // Format the frequent itemsets based on the number of items
    const formatted = results.map(result => {
      const formattedItemsets = {};
      result.frequent_itemsets.forEach(itemset => {
        const numItems = itemset.length;

        if (!formattedItemsets[numItems]) {
          formattedItemsets[numItems] = [];
        }

        formattedItemsets[numItems].push(itemset.join(', '));
      });

      return {
        support: result.support,
        confidence: result.confidence,
        totalRules: result.total_rules,
        itemsets: formattedItemsets,
      };
    });

    setFormattedData(formatted);
  }, [results]);

  return (
    <div>
      {formattedData.map((result, index) => (
        <div key={index}>
          <h3>Support: {result.support}, Confidence: {result.confidence}, Total Rules: {result.totalRules}</h3>
          {Object.keys(result.itemsets).map(numItems => (
            <div key={numItems}>
              <h4>{numItems}-Item Itemsets</h4>
              <ul>
                {result.itemsets[numItems].map((itemset, idx) => (
                  <li key={idx}>{itemset}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

export default AssociationRulesTable;
