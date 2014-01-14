import csv
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

ANNOTATION_PATH = 'user_annotations.csv'
IMAGE_DIR = '/Users/dhaas/Code/projects/Art_Vision/Picasso/People/'
PCT_IMAGES_TO_PLOT = 20

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

if __name__ == '__main__':
    user_annotations = parse_annotations(ANNOTATION_PATH)
    out = {}
    for imgid, annotation_list in user_annotations.iteritems():
        clustered_annotations = cluster_annotations(annotation_list)
        out[imgid] = clustered_annotations
        if random.random() < PCT_IMAGES_TO_PLOT / 100.0:
            plotted = plot_annotations(imgid, clustered_annotations)        
