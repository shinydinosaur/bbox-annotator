import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from annotation import annotations_by_user

def cluster_annotations(annotations):
    X = [annotation.feature_vector() for annotation in annotations]
    by_user = annotations_by_user(zip(annotations, range(len(annotations))))
    num_per_user = [len(by_user[user]) for user in by_user]

    # try DBSCAN clustering
    # hack because annotations_by_user expects clusterids as well.
    #model = DBSCAN(eps=49.0, # TUNE THIS
    #               min_samples=len(by_user)/4.0)

    # try k-means, using the median annotations per user as the number of
    # clusters, and a random users' annotations as initial points.
    n_clusters=np.round(np.median(num_per_user))
    potential_user_annotations = [annotation_list
                                  for annotation_list in by_user.values()
                                  if len(annotation_list) == n_clusters]
    try:
        init = np.array(
            [anno_tuple[0].feature_vector()
             for anno_tuple in random.choice(potential_user_annotations)])
    except IndexError:
        init = 'k-means++'

    model = KMeans(n_clusters=int(np.median(num_per_user)),
                   init=init)

    cluster_assignments = model.fit_predict(X)
    return [(annotations[i], cluster_index)
            for i, cluster_index in enumerate(cluster_assignments)]

def assign_clusters(annotations, cluster_medians):
    clustered_annotations = []
    for annotation in annotations:
        closest_id = -1
        closest_dist = np.Inf
        for cluster_id, cluster_median in cluster_medians.iteritems():
            dist = annotation.distance_to(cluster_median)
            if dist < closest_dist:
                closest_id = cluster_id
                closest_dist = dist
        clustered_annotations.append((annotation, closest_id))
    return clustered_annotations
