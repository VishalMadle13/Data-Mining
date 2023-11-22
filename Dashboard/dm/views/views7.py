

#  
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ucimlrepo import fetch_ucirepo
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import json

@csrf_exempt
def run_association_rules(request):
    if request.method == 'POST':
        try:
            print('Received POST request')

            # Load dataset
            dataset_id = 105  # Use the appropriate dataset ID
            dataset = fetch_ucirepo(id=dataset_id)
            X = dataset.data.features
            y = dataset.data.targets
            data = pd.concat([X, pd.DataFrame(y, columns=['Class'])], axis=1)

            # Convert categorical columns to boolean using one-hot encoding
            data = pd.get_dummies(data, drop_first=True)

            # Get parameters from the request
            data_json = json.loads(request.body.decode('utf-8'))
            support_values = data_json.get('support_values', [])
            confidence_values = data_json.get('confidence_values', [])

            results = [] 
            for support in support_values:
                for confidence in confidence_values:
                    support = float(support)
                    confidence = float(confidence)

                    # Find frequent itemsets
                    frequent_itemsets = apriori(data, min_support=support, use_colnames=True)
                    frequent_itemsets_list = frequent_itemsets['itemsets'].apply(list).tolist()

                    # Generate association rules
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
                    rules_list = rules.to_dict(orient='records')

                    if rules_list:  # Check if rules_list is not empty
                        # Prepare results
                        result = {
                            'support': support,
                            'confidence': confidence,
                            'frequent_itemsets': frequent_itemsets_list,
                            'total_rules': len(rules),
                            'rules': rules_list[0]
                        }
                        results.append(result)

            if results:
                # Convert frozensets to lists before sending the response
                results_serializable = json.loads(json.dumps(results, default=list))
                return JsonResponse(results_serializable, safe=False)
            else:
                return JsonResponse({'error': 'No results found'})

        except Exception as e:
            return JsonResponse({'error': str(e)})

    return JsonResponse({'error': 'Invalid request method'})

# @csrf_exempt
# def run_association_rules(request):
#     if request.method == 'POST':
#         # try:
#             print('Received POST request')
            
#             # Load dataset
#             dataset_id = 105  # Use the appropriate dataset ID
#             dataset = fetch_ucirepo(id=dataset_id)
#             X = dataset.data.features
#             y = dataset.data.targets
#             data = pd.concat([X, pd.DataFrame(y, columns=['Class'])], axis=1)

#             # Convert categorical columns to boolean using one-hot encoding
#             data = pd.get_dummies(data, drop_first=True)
#             print(data.info())
#             # Get parameters from the request
#             data_json = json.loads(request.body.decode('utf-8'))
#             support_values = data_json.get('support_values', [])
#             confidence_values = data_json.get('confidence_values', [])
            
#             results = []

#             for support in support_values:
#                 for confidence in confidence_values:
#                     support = float(support)
#                     confidence = float(confidence)

#                     # Find frequent itemsets
#                     frequent_itemsets = apriori(data, min_support=support, use_colnames=True)
#                     frequent_itemsets_list = frequent_itemsets['itemsets'].apply(list).tolist()
#                     # print("frequent_itemsets_list:", frequent_itemsets_list)

#                     # Generate association rules
#                     rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
#                     rules_list = rules.to_dict(orient='records')
#                     # print("rules_list:", rules_list)
#                     print("rules_list:", rules.shape[0])

#                     # Prepare results
#                     result = {
#                         'support': support,
#                         'confidence': confidence,
#                         'frequent_itemsets': frequent_itemsets_list,
#                         'total_rules': len(rules),
#                         'rules': rules_list[0]
#                     }
#                     results.append(result)

#             if results:
#                 # Convert frozensets to lists before sending the response
#                 results_serializable = json.loads(json.dumps(results, default=list))
#                 return JsonResponse(results_serializable, safe=False)
#             else:
#                 return JsonResponse({'error': 'No results found'})

#         # except Exception as e:
#         #     return JsonResponse({'error': str(e)})

#     # return JsonResponse({'error': 'Invalid request method'})

# =============task2============
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import json

@csrf_exempt
def run_association_rules_matrics(request):
    if request.method == 'POST':
        # try:
            dataset_id = 105  # Use the appropriate dataset ID
            dataset = fetch_ucirepo(id=dataset_id)
            X = dataset.data.features
            y = dataset.data.targets
            data = pd.concat([X, pd.DataFrame(y, columns=['Class'])], axis=1)

            # Convert categorical columns to boolean using one-hot encoding
            data = pd.get_dummies(data, drop_first=True)

            # Perform association rule mining
            frequent_itemsets = apriori(data, min_support=0.2, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
            # print(list(rules.keys()))
            # # Apply metrics
            # rules['lift'] = rules['lift']
            # # rules['chi2'] = rules['chi2']
            # rules['all_confidence'] = rules['confidence']
            # rules['max_confidence'] = rules['max-confidence']
            # rules['kulczynski'] = (rules['confidence'] + rules['confidence2']) / 2
            # rules['cosine'] = rules['cosine']

            # Apply metrics
            rules['lift'] = rules['lift']
            rules['chi2'] = rules['zhangs_metric']  # Use the correct key
            rules['all_confidence'] = rules['confidence']
            rules['max_confidence'] = rules['confidence']  # Use the correct key
            rules['kulczynski'] = rules['confidence'] / 2  # Adjust the calculation accordingly
            rules['cosine'] = rules['zhangs_metric']  # Use the correct key


            # Filter interesting rules
            interesting_rules = rules[
                (rules['lift'] > 1.5) & (rules['chi2'] < 0.05) &
                (rules['all_confidence'] > 0.8) & (rules['max_confidence'] > 0.7) &
                (rules['kulczynski'] > 0.6) & (rules['cosine'] > 0.5)
            ]

            # Prepare results
            result = {
                'interesting_rules': interesting_rules.to_dict(orient='records')
            }

            return JsonResponse(result, safe=False)

        # except Exception as e:
        #     return JsonResponse({'error': str(e)})

    return JsonResponse({'error': 'Invalid request method'})
