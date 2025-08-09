from calibrate.cluster_model import ClusterFactory, ClusteringOptimizer
from collections import defaultdict
from pprint import pprint
sector = "Information Technology"
start_date = 20100101
end_date = 20170101

results = defaultdict()
for method in ["kmeans", "gmm", "hierarchical", "spectral"]:
    print(f"\n--- Testing {method.upper()} ---")
    model = ClusterFactory.create(method, sector, start_date, end_date)
    optimizer = ClusteringOptimizer(model)
    best_params, best_score, all_results = optimizer.search()
    print(f"Best_parameters: {best_params}")
    print(f"custom score: {best_score}")
    results[method] = (best_params, best_score)

pprint(results)

    
    

