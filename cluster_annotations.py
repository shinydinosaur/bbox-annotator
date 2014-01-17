import csv
import random
import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

ANNOTATION_PATH = 'user_annotations_new.csv'
IMAGE_DIR = '/Users/dhaas/Code/projects/Art_Vision/Picasso/People/'
PCT_IMAGES_TO_PLOT = 0

COLOR_MAP = {
    -1: 'k',
    0: 'r', 
    1: 'b', 
    2: 'g', 
    3: 'c',
    4: 'm',
    5: 'y',
    6: 'w'
}

class Annotation(object):
    def __init__(self, raw_row):
        self.imgid = raw_row['imgid'].split("/")[-1]
        self.userid = int(raw_row['userid'])
        self.left = float(raw_row['bbox_left'])
        self.top = float(raw_row['bbox_top'])
        self.width = float(raw_row['bbox_width'])
        self.height = float(raw_row['bbox_height'])

    def area(self):
        return self.width * self.height

    def intersects(self, other_annotation):
        # Self is fully to the right of other_annotation.
        if self.left > other_annotation.left + other_annotation.width:
            return False

        # Self is fully to the left of other_annotation.
        if self.left + self.width < other_annotation.left:
            return False

        # Self is fully below other_annotation.
        if self.top > other_annotation.top + other_annotation.height:
            return False

        # Self is fully above other_annotation.
        if self.top + self.height < other_annotation.top:
            return False

        # There must be overlap.
        return True

    def intersection_size(self, other_annotation):
        if not self.intersects(other_annotation):
            return 0.0
        intersection_left = max((self.left, other_annotation.left))
        intersection_top = max((self.top, other_annotation.top))
        intersection_right = min((self.left + self.width, other_annotation.left + other_annotation.width))
        intersection_bottom = min((self.top + self.height, other_annotation.top + other_annotation.height))
        intersection_width = intersection_right - intersection_left
        intersection_height = intersection_bottom - intersection_top
        return intersection_width * intersection_height

    def union_size(self, other_annotation):
        return self.area() + other_annotation.area() - self.intersection_size(other_annotation)

def parse_annotations(annotation_file):
    annotations = {}
    with open(annotation_file, 'rb') as anno_file:
        reader = csv.DictReader(anno_file, delimiter=',')
        for row in reader:
            annotation = Annotation(row)
            if annotation.imgid not in annotations:
                annotations[annotation.imgid] = []
            annotations[annotation.imgid].append(annotation)
    return annotations

def cluster_annotations(annotations):
    # try clustering on the centroids of the boxes
    # X = np.array([((annotation.left + annotation.width) / 2.0, 
    #                (annotation.top + annotation.height) / 2.0)
    #               for annotation in annotations])

    # try clustering on all corners of the boxes
    X = np.array([(annotation.left, 
                   annotation.left + annotation.width, 
                   annotation.top,
                   annotation.top + annotation.height)
                  for annotation in annotations])

    # try k-means with 2 clusters
    #model = KMeans(n_clusters=2, init='random')
    
    # try DBSCAN clustering 
    model = DBSCAN(eps=50.0, # TUNE THIS
                   # TODO: some function of the number of users who annotated this image
                   min_samples = 4)
    
    # try affinity propagation
    # model = AffinityPropagation(damping=0.95) # TUNE ME

    # TODO(dhaas): figure out n_clusters, modify init if we have more info.
    cluster_assignments = model.fit_predict(X)
    return [(annotations[i], cluster_index) 
            for i, cluster_index in enumerate(cluster_assignments)]

def plot_annotations(imgid, clustered_annotations):
    try:
        img = mpimg.imread(IMAGE_DIR + imgid)
        pass
    except IOError:
        print "WARNING: couldn't find image '%s'" % imgid
        return False

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(img)
    for annotation, cluster_id in clustered_annotations:
        color = COLOR_MAP[cluster_id]
        rect = Rectangle((annotation.left, annotation.top),
                         annotation.width, annotation.height, fill=False, color=color)
        ax.add_patch(rect)
    plt.show()
    return True

def annotations_by_user(annotations, userids=None):
    by_user = { userid: [] for userid in userids } if userids else {}
    for annotation, cluster_id in annotations:
        if userids is None or annotation.userid in userids:
            if annotation.userid not in by_user:
                by_user[annotation.userid] = []
            by_user[annotation.userid].append((annotation, cluster_id))
    return by_user

def annotations_by_cluster(annotations, clusterids=None):
    by_cluster = {}
    for annotation, cluster_id in annotations:
        if clusterids is None or cluster_id in clusterids:
            if cluster_id not in by_cluster:
                by_cluster[cluster_id] = []
            by_cluster[cluster_id].append((annotation, cluster_id))
    return by_cluster        

def overlap_error_score(user1_annotation, user2_annotation):
    # compute as 1 - box intersection / box union (this is bi-directional)
    return 1.0 - (user1_annotation.intersection_size(user2_annotation) / user1_annotation.union_size(user2_annotation))

if __name__ == '__main__':
    annotations = parse_annotations(ANNOTATION_PATH)
    scores = {}
    fps = {}
    fns = {}
    for imgid, annotation_list in annotations.iteritems():
        
        # cluster the annotations
        clustered_annotations = cluster_annotations(annotation_list)
        if random.random() < PCT_IMAGES_TO_PLOT / 100.0:
            plotted = plot_annotations(imgid, clustered_annotations)        
        
        # compute pairwise error scores for each figure
        by_cluster = annotations_by_cluster(clustered_annotations)
        all_users = annotations_by_user(clustered_annotations).keys()
        overlap_scores = {}
        false_positives = {}
        false_negatives = {}
        for user1, user2 in combinations(all_users, 2):
            dict_key = str(sorted((user1, user2))) # allow bidirectional lookup
            u1_key = str(user1)
            u2_key = str(user2)
            overlap_scores[dict_key] = []
            false_positives[u1_key] = 0
            false_positives[u2_key] = 0
            false_negatives[u1_key] = []
            false_negatives[u2_key] = []

            for cluster_id, annotation_list in by_cluster.iteritems():
                u1_false_positives = u2_false_positives = 0
                u1_false_negatives = u2_false_negatives = 0
                user_annotations = annotations_by_user(annotation_list, [user1, user2])

                # Handle the outlier cluster--these are all false positives.
                if cluster_id == -1: 
                    u1_false_positives += len(user_annotations[user1])
                    u2_false_positives += len(user_annotations[user2])
                    false_positives[u1_key] = u1_false_positives
                    false_positives[u2_key] = u2_false_positives
                    continue

                # Handle False negatives: users didn't draw a box for this figure.
                overlap_possible = True
                if len(user_annotations[user1]) == 0:
                    u1_false_negatives += 1 
                    overlap_possible = False
                if len(user_annotations[user2]) == 0:
                    u2_false_negatives += 1
                    overlap_possible = False
                false_negatives[u1_key].append(u1_false_negatives)
                false_negatives[u2_key].append(u2_false_negatives)
                if not overlap_possible:
                    continue

                # Otherwise, overlap scores
                if len(user_annotations[user1]) != 1:
                    # Either user accidentally drew too many rectangles or algorithm
                    # misclassified a rectangle because it was too close to the cluster.
                    # Either way, pick a random rectangle to evaluate.
                    user_annotations[user1][0] = random.choice(user_annotations[user1])
                if len(user_annotations[user2]) != 1:
                    user_annotations[user2][0] = random.choice(user_annotations[user2])
                overlap_scores[dict_key].append(
                    overlap_error_score(
                        user_annotations[user1][0][0], 
                        user_annotations[user2][0][0]))

            # Aggregate False Positives and False Negatives over the image.
            # False Positives are fine as is.
            # Score False negatives as #FN / #Clusters.
            u1_fn_scores = false_negatives[u1_key]
            u2_fn_scores = false_negatives[u2_key]            
            false_negatives[u1_key] = np.mean(u1_fn_scores) if u1_fn_scores else 0.0
            false_negatives[u2_key] = np.mean(u2_fn_scores) if u2_fn_scores else 0.0

        # Average overlap scores per user pair per image
        scores[imgid] = { key : np.mean(overlap_score_list) for key, overlap_score_list in overlap_scores.iteritems() if overlap_score_list }
        fps[imgid] = false_positives
        fns[imgid] = false_negatives

    # plot the distribution of error between pairs of users on the same image
    plt.figure()
    data_same = [error_score for img_dict in scores.itervalues() for error_score in img_dict.itervalues()]
    n, bins, patches = plt.hist(data_same, 20, histtype='bar', normed=True)
    plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
    plt.xlim([0,1])
    plt.title("Overlap Error scores (same images)")
    plt.show()

    # Draw a boxplot by image
    # use ~1/10 of the images so plot is readable
    boxplot_data = [img_dict.values() for img_dict in scores.values()]
    num_images = len(boxplot_data)
    sample_indexes = sorted(random.sample(range(num_images), num_images / 10))
    boxplot_data = [img_scores for i, img_scores in enumerate(boxplot_data) if i in sample_indexes]
    img_labels = ["Image %s" % str(i) for i in sample_indexes]
    fig = plt.figure()
    ax = plt.gca()
    plt.boxplot(boxplot_data, notch=False, sym='')
    xtickNames = plt.setp(ax, xticklabels=img_labels)
    plt.setp(xtickNames, rotation=45, fontsize=8)
    plt.show()

    # plot false positives
    plt.figure()
    data_fp = [img_dict.values() for i, img_dict in enumerate(fps.values()) if i in sample_indexes]
    ax = plt.gca()
    plt.boxplot(data_fp, sym='')
    xtickNames = plt.setp(ax, xticklabels=img_labels)
    plt.setp(xtickNames, rotation=45, fontsize=8)
    plt.title("False Positive Scores")
    plt.ylim([-1, max([max(scores) for scores in data_fp]) + 2])
    plt.show()

    # plot false negatives
    plt.figure()
    data_fn = [img_dict.values() for i, img_dict in enumerate(fns.values()) if i in sample_indexes]
    ax = plt.gca()
    plt.boxplot(data_fn, sym='')
    xtickNames = plt.setp(ax, xticklabels=img_labels)
    plt.setp(xtickNames, rotation=45, fontsize=8)
    plt.title("False Negative Scores")
    plt.ylim([-1, 2])
    plt.show()

    exit() # Don't wait forever for the last bit!

    # plot the distribution of error scores between pairs of users on different images
    all_annotations = [annotation for user_annotation_list in annotations.itervalues() for annotation in user_annotation_list]
    data_different = []
    for anno1, anno2 in combinations(all_annotations, 2):
        if anno1.imgid != anno2.imgid:
            data_different.append(overlap_error_score(anno1, anno2))
    plt.figure()
    n, bins, patches = plt.hist(data_different, 20, histtype='bar', normed=True)
    plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
    plt.xlim([0,1])
    plt.title("Overlap Error scores (different images)")
    plt.show()
        
