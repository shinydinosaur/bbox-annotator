import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from sklearn.metrics import auc

from annotation import annotations_by_cluster, median_annotation
from config import *

def plot_prec_recall(tps, fps, npos, title, filename=None, show=True):
    recall = [float(tp) / float(npos) for tp, npos in zip(tps, npos)]
    precision = [float(tp) / (float(tp) + float(fp))
                 for tp, fp in zip(tps, fps)]
    auc_score = auc(recall, precision, reorder=True)
    plt.figure()
    plt.plot(recall, precision, '-')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title(title + " AUC: " + str(auc_score))
    if filename:
        plt.savefig(os.path.join(RESULTS_PATH, filename))
    if show:
        plt.show()

def plot_distribution(X, title, nbins=10, set_lims=True, normed=True,
                      filename=None, show=True):
    plt.figure()
    n, bins, patches = plt.hist(X, nbins, histtype='bar', normed=normed)
    plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
    # patches[0].set_facecolor('r')
    if set_lims:
        plt.xlim([0,1])
    plt.title(title)
    if filename:
        plt.savefig(os.path.join(RESULTS_PATH, filename))
    if show:
        plt.show()

def plot_annotations(imgid, clustered_annotations):
    try:
        img = mpimg.imread(IMAGE_DIR + imgid + ".JPG")
        pass
    except IOError:
        print "WARNING: couldn't find image '%s'" % imgid
        return False

    # Plot all annotations
    fig = plt.figure()
    ax = plt.gca()
    plt.imshow(img)
    for annotation, cluster_id in clustered_annotations:
        color = COLOR_MAP[cluster_id]
        rect = Rectangle((annotation.left, annotation.top),
                         annotation.width, annotation.height,
                         fill=False, color=color)
        ax.add_patch(rect)
    plt.show()

    # Plot median human and computer annotations
    by_cluster = annotations_by_cluster(clustered_annotations)
    all_medians = { clusterid :
                    (median_annotation(annotations),
                     median_annotation([annotation for annotation in annotations
                                        if annotation[0].is_human]),
                     median_annotation([annotation for annotation in annotations
                                        if not annotation[0].is_human]))
                      for clusterid, annotations in by_cluster.iteritems() }
    plt.figure()
    ax = plt.gca()
    plt.imshow(img)
    for clusterid, medians in all_medians.iteritems():
        color = COLOR_MAP[clusterid]
        for median in medians:
            if median is None: continue
            rect = Rectangle((median.left, median.top),
                             median.width, median.height, fill=False, color=color)
            ax.add_patch(rect)
    plt.show()

    # Affine transform image to consistent shape and plot again.
    from skimage.transform import AffineTransform, warp

    # calculate scale and coreners
    row_scale = float(img.shape[0]) / 400.0
    col_scale = float(img.shape[1]) / 400.0
    src_corners = np.array([[1, 1], [1, 400.0], [400.0, 400.0]]) - 1
    dst_corners = np.zeros(src_corners.shape, dtype=np.double)
    # take into account that 0th pixel is at position (0.5, 0.5)
    dst_corners[:, 0] = col_scale * (src_corners[:, 0] + 0.5) - 0.5
    dst_corners[:, 1] = row_scale * (src_corners[:, 1] + 0.5) - 0.5

    # do the transformation
    tform = AffineTransform()
    tform.estimate(src_corners, dst_corners)
    resized = warp(img, tform, output_shape=[400.0, 400.0], order=1,
                   mode='constant', cval=0)

    # plot the transformed image
    plt.figure()
    ax = plt.gca()
    plt.imshow(resized)
    for clusterid, medians in all_medians.iteritems():
        color = COLOR_MAP[clusterid]

        for median in medians:
            if median is None: continue

            # apply the transformation to each rectangle
            corners = np.array([[median.left, median.top],
                                [median.left + median.width,
                                 median.top + median.height]])
            new_corners = tform.inverse(corners)
            rect = Rectangle(new_corners[0, :],
                             new_corners[1,0] - new_corners[0,0],
                             new_corners[1,1] - new_corners[0,1],
                             fill=False, color=color)
            ax.add_patch(rect)
    plt.show()
    return True
