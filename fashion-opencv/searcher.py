import numpy as np
from scipy.spatial.distance import cosine, euclidean

class Searcher:
    def __init__(self, indexPath):
        # store our index path
        self.indexPath = indexPath
        # Load the .npz file
        data = np.load(self.indexPath, allow_pickle=True)
        self.image_names = data['image_names']
        self.features = data['features']
        
    def search(self, queryFeatures, limit=10, distance_metric='chi2'):
        # initialize our dictionary of results
        results = {}

        # loop over the features in the loaded .npz file
        for idx, feature in enumerate(self.features):
            # compute the distance between the features in our index using the chosen metric
            if distance_metric == 'chi2':
                d = self.chi2_distance(feature, queryFeatures)
            elif distance_metric == 'cosine':
                d = self.cosine_distance(feature, queryFeatures)
            elif distance_metric == 'euclidean':
                d = self.euclidean_distance(feature, queryFeatures)
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")

            # update the results dictionary with the image name and distance
            results[self.image_names[idx]] = d

        # sort our results, so that the smaller distances (more similar images) are at the front
        results = sorted([(v, k) for (k, v) in results.items()])
        # return the top 'limit' results
        return results[:limit]

    def chi2_distance(self, histA, histB, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
        return d

    def cosine_distance(self, histA, histB):
        # compute the cosine distance (1 - cosine similarity)
        return cosine(histA, histB)

    def euclidean_distance(self, histA, histB):
        # compute the Euclidean distance
        return euclidean(histA, histB)
