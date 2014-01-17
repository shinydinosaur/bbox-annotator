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
    by_user = {}
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
    user_annotations = parse_annotations(ANNOTATION_PATH)
    out = {}
    scores = {}
    for imgid, annotation_list in user_annotations.iteritems():
        
        # cluster the annotations
        clustered_annotations = cluster_annotations(annotation_list)
        out[imgid] = clustered_annotations
        if random.random() < PCT_IMAGES_TO_PLOT / 100.0:
            plotted = plot_annotations(imgid, clustered_annotations)        
        
        # compute pairwise error scores for each figure
        overlap_scores = {}
        for clusterid, annotation_list in annotations_by_cluster(clustered_annotations).iteritems():

            # ignore the outlier cluster
            if clusterid == -1:
                continue

            by_user = annotations_by_user(annotation_list)
            userids = by_user.keys()
            for user1, user2 in combinations(userids, 2):
                user1_annotations = by_user[user1]
                user2_annotations = by_user[user2]
                if len(user1_annotations) != 1:
                    # print "WARNING: user %d had multiple annotations of person %d in image %s" % (user1, clusterid, imgid)
                    user1_annotations[0] = random.choice(user1_annotations)

                if len(user2_annotations) != 1:
                    # print "WARNING: user %d had multiple annotations of person %d in image %s" % (user2, clusterid, imgid)
                    user2_annotations[0] = random.choice(user2_annotations)

                overlap_score = overlap_error_score(user1_annotations[0][0], user2_annotations[0][0])
                dict_key = str(sorted((user1, user2))) # allow bidirectional lookup
                if dict_key not in overlap_scores:
                    overlap_scores[dict_key] = []
                overlap_scores[dict_key].append(overlap_score)

        # Average overlap scores per user pair per image
        scores[imgid] = { key : np.mean(overlap_score_list) for key, overlap_score_list in overlap_scores.iteritems() }

    # plot the distribution of error between pairs of users on the same image
    plt.figure()
    data_same = [error_score for img_dict in scores.itervalues() for error_score in img_dict.itervalues()]
    n, bins, patches = plt.hist(data_same, 20, histtype='bar', normed=True)
    plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
    plt.xlim([0,1])
    plt.title("Overlap Error scores (same images)")
    plt.show()

    # plot the distribution of error scores between pairs of users on different images
    all_annotations = [user_annotation for user_annotation_list in user_annotations.itervalues() for user_annotation in user_annotation_list]
    data_different = []
    for anno1, anno2 in combinations(all_annotations, 2):
        if anno1.imgid != anno2.imgid:
            try:
                data_different.append(overlap_error_score(anno1, anno2))
            except ZeroDivisionError:
                import pdb;pdb.set_trace()
    plt.figure()
    n, bins, patches = plt.hist(data_different, 20, histtype='bar', normed=True)
    plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
    plt.xlim([0,1])
    plt.title("Overlap Error scores (different images)")
    plt.show()
        
