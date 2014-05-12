import csv
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from annotation import (Annotation,
                        median_annotation,
                        annotations_by_user,
                        annotations_by_cluster,
                        annotations_by_humanity,
                        annotations_by_confidence,
                        unique_users,
                        split_annotations)
from buckets import assign_buckets
from cluster import cluster_annotations, assign_clusters
from config import *
from loader import (parse_csv_annotations,
                    annotation_list_to_dict,
                    load_algorithm)
from plot import plot_annotations, plot_distribution, plot_prec_recall

EVAL_CACHE = {}

def save_medians(clustered_annotations):
    by_cluster = annotations_by_cluster(clustered_annotations)
    medians = { clusterid : median_annotation(annotations)
                for clusterid, annotations in by_cluster.iteritems() }
    with open(MEDIAN_PATH, 'ab') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['imgid', 'clusterid',
                                            'left', 'top', 'width', 'height'])
        writer.writerows([{'imgid': imgid,
                           'clusterid': clusterid,
                           'left': anno.left,
                           'top': anno.top,
                           'width': anno.width,
                           'height': anno.height}
                          for clusterid, anno in medians.iteritems()])
    return medians


def overlap_error_score(user1_annotation, user2_annotation):
    # compute as 1 - box intersection / box union (this is bi-directional)
    return 1.0 - (user1_annotation.intersection_size(user2_annotation)
                  / user1_annotation.union_size(user2_annotation))

def f_measure(num_tps, num_fps, num_pos):
    return (2.0 * float(num_tps) /
            (float(num_tps) + float(num_fps) + float(num_pos)))

def eval_comp(comp_annos, gt_annos, confidences):
    tps = []
    fps = []
    sorted_annos = annotations_by_confidence(comp_annos)
    tps_raw, fps_raw, npos = eval_user(sorted_annos, gt_annos)
    if len(comp_annos) == 0:
        return ([0] * len(confidences),
                [0] * len(confidences),
                [npos] * len(confidences))

    cur_idx = 0
    tps_sofar = 0
    fps_sofar = 0
    for conf in confidences:
        new_tps = 0
        new_fps = 0
        if cur_idx < len(sorted_annos):
            while conf <= sorted_annos[cur_idx].confidence:
                new_tps += tps_raw[cur_idx]
                new_fps += fps_raw[cur_idx]
                cur_idx += 1
                if cur_idx >= len(sorted_annos):
                    break
        tps_sofar += new_tps
        fps_sofar += new_fps
        tps.append(tps_sofar)
        fps.append(fps_sofar)
    return (tps, fps, [npos] * len(confidences))

#    tps = []
#    fps = []
#    npos = []
#    last_annos = []
#    for confidence in confidences:
#        cur_annos = [anno for anno in comp_annos
#                     if anno.confidence >= confidence]
#        if not cur_annos:
#            tps.append(0)
#            fps.append(0)
#        elif cur_annos == last_annos:
#            tps.append(tps[-1])
#            fps.append(fps[-1])
#        else:
#            conf_tps, conf_fps, conf_npos = eval_user(cur_annos, gt_annos)
#            tps.append(conf_tps)
#            fps.append(conf_fps)
#            if not npos:
#                npos = [conf_npos]*len(confidences)
#        last_annos = cur_annos
#    if not npos: # computer had no detections at any confidence
#        num_pos = len(annotations_by_cluster(cluster_annotations(gt_annos)))
#        npos = [num_pos]*len(confidences)
#    return tps, fps, npos

def eval_user(user_annos, gt_annos):
    tps = []
    fps = []

    clusters = annotations_by_cluster(cluster_annotations(gt_annos))
    gt_boxes = [median_annotation(annotations)
                for annotations in clusters.itervalues()]
    npos = len(gt_boxes)

    # Check the user's annotations against ground truth.
    hits = [False]*npos
    for user_anno in user_annos:
        best_box_index = -1
        best_overlap_score = 2.0
        for i, gt_box in enumerate(gt_boxes):
            cur_overlap_score = overlap_error_score(user_anno, gt_box)
            if cur_overlap_score < best_overlap_score:
                best_overlap_score = cur_overlap_score
                best_box_index = i

        if best_overlap_score < OVERLAP_THRESH: # Hit!
            if hits[best_box_index]:
                fps.append(1) # duplicate!
                tps.append(0)
            else:
                tps.append(1)
                fps.append(0)
                hits[best_box_index] = True
        else:
            fps.append(1)
            tps.append(0)

    return (tps, fps, npos)

def eval_algorithms(bucket, human_annotations, algorithms):
    users = unique_users(human_annotations.values())
    for algorithm in ALGORITHMS:
        users.add(algorithm['name'])

    # Iterate over the users, including the computers
    user_fmeasures = []
    for userid in users:
        if userid in algorithms:
            comp_annos = algorithms[userid][0]
            confs = algorithms[userid][1]
            nconfs = len(confs)
            tps = [0] * nconfs
            fps = [0] * nconfs
            npos = [0] * nconfs
        else:
            tps = fps = npos = 0

        # Iterate over the images
        imgidx = -1
        num_imgs = len(bucket.imgids)
        for imgid, annotation_list in human_annotations.iteritems():
            if imgid not in bucket.imgids:
                continue

            # Computers: evaluate the computer annotations against all
            # user annotations.
            if userid in algorithms:
                imgidx += 1
                if imgidx % 20 == 0:
                    print ("Completed %d/%d images for user %s"
                           % (imgidx, num_imgs, userid))
                image_tps, image_fps, image_npos = eval_comp(
                    comp_annos[imgid],
                    annotation_list,
                    confs)
                tps = [x + y for x, y in zip(image_tps, tps)]
                fps = [x + y for x, y in zip(image_fps, fps)]
                npos = [x + y for x, y in zip(image_npos, npos)]

            # Separate our user's annotations from the rest and evaluate the
            # user against all other annotations.
            else:
                user_annos, gt_annos = split_annotations(annotation_list,
                                                         userid=userid)

                if not user_annos: # user didn't annotate this image
                    continue
                image_tps, image_fps, image_npos = eval_user(
                    user_annos, gt_annos)
                tps += sum(image_tps)
                fps += sum(image_fps)
                npos += image_npos

        # Compute a user-specific F-measure
        if userid in algorithms:
            fmeasures = [f_measure(tp, fp, num_p)
                         for tp, fp, num_p in zip(tps, fps, npos)]
            max_fmeasure = max(fmeasures)
            print "Max f_measure for user", userid, ":", max_fmeasure
            plot_prec_recall(tps, fps, npos,
                             ("Precision-Recall Curve for user " + userid
                              + " (Method " + bucket.method + " Bucket "
                              + str(bucket.name) + ")"),
                             filename="%s_prec_recall_%s_%d.png" % (
                                 userid, bucket.method, bucket.name),
                             show=False)
        else:
            # user may not have annotated anything in this bucket
            if npos == 0:
                print "F_measure for user", userid, ": Not Enough Data"
            else:
                fm = f_measure(tps, fps, npos)
                user_fmeasures.append(fm)
                print "F_measure for user", userid, ":", fm

    # Compute an average F-measure
    mean_fmeasure = np.mean(user_fmeasures)
    median_fmeasure = np.median(user_fmeasures)
    print "Mean human f_measure (Method %s, bucket %d):" % (
        bucket.method, bucket.name), mean_fmeasure
    print "Median human f_measure (Method %s, bucket %d):" % (
        bucket.method, bucket.name), median_fmeasure

    # Plot the F-measure distribution
    plot_distribution(user_fmeasures,
                      "User F-measure distribution (Method %s, bucket %d)" % (
                          bucket.method, bucket.name),
                      filename="user_fmeasure_dist_%s_%d" % (
                          bucket.method.lower(), bucket.name),
                      show=False)

if __name__ == '__main__':
    human_annotations = annotation_list_to_dict(
        parse_csv_annotations(HUMAN_ANNOTATION_PATH))

    algorithms = {}
    for algorithm in ALGORITHMS:
        load_algorithm(algorithm, CONFIDENCE_THRESH, algorithms)

    (difficulty_buckets,
     time_equiwidth_buckets,
     time_equidepth_buckets,
     all_bucket) = assign_buckets()

    for bucket in (all_bucket + difficulty_buckets
                   + time_equiwidth_buckets + time_equidepth_buckets):
        print "Evaluating bucket %d (method %s)" % (bucket.name, bucket.method)
        eval_algorithms(bucket, human_annotations, algorithms)
