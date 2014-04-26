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
                        unique_users,
                        split_annotations)
from cluster import cluster_annotations, assign_clusters
from config import *
from loader import (parse_csv_annotations,
                    parse_matlab_annotations,
                    annotation_list_to_dict)
from plot import plot_annotations, plot_distribution


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
    npos = []
    last_annos = []
    for confidence in confidences:
        cur_annos = [anno for anno in comp_annos
                     if anno.confidence >= confidence]
        if not cur_annos:
            tps.append(0)
            fps.append(0)
        elif cur_annos == last_annos:
            tps.append(tps[-1])
            fps.append(fps[-1])
        else:
            conf_tps, conf_fps, conf_npos = eval_user(cur_annos, gt_annos)
            tps.append(conf_tps)
            fps.append(conf_fps)
            if not npos:
                npos = [conf_npos]*len(confidences)
        last_annos = cur_annos
    if not npos: # computer had no detections at any confidence
        num_pos = len(annotations_by_cluster(cluster_annotations(gt_annos)))
        npos = [num_pos]*len(confidences)
    return tps, fps, npos

def eval_user(user_annos, gt_annos):
    tps = fps = npos = 0

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
                fps += 1 # duplicate!
            else:
                tps += 1
                hits[best_box_index] = True
        else:
            fps += 1

    return (tps, fps, npos)

if __name__ == '__main__':
    human_annotations = annotation_list_to_dict(
        parse_csv_annotations(HUMAN_ANNOTATION_PATH))
    poselets_annos, poselets_confidences = parse_matlab_annotations(
        POSELETS_ANNOTATION_PATH,
        POSELETS_USERID,
        min_conf=CONFIDENCE_THRESH)
    poselets_annotations = annotation_list_to_dict(poselets_annos)

    # Iterate over the users, including the computers
    user_fmeasures = []
    users = unique_users(human_annotations.values())
    users.update(COMP_USERIDS)
    for userid in users:
        if userid in COMP_USERIDS:
            if userid == POSELETS_USERID:
                confs = poselets_confidences
                comp_annos = poselets_annotations
            nconfs = len(confs)
            tps = [0] * nconfs
            fps = [0] * nconfs
            npos = [0] * nconfs
        else:
            tps = fps = npos = 0

        # Iterate over the images
        imgidx = -1
        num_imgs = len(human_annotations.keys())
        for imgid, annotation_list in human_annotations.iteritems():

            # Computers: evaluate the computer annotations against all
            # user annotations.
            if userid in COMP_USERIDS:
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
                image_tps, image_fps, image_npos = eval_user(
                    user_annos, gt_annos)
                tps += image_tps
                fps += image_fps
                npos += image_npos

        # Compute a user-specific F-measure
        if userid in COMP_USERIDS:
            # TODO: plot precision-recall curve?
            fmeasures = [f_measure(tp, fp, num_p)
                         for tp, fp, num_p in zip(tps, fps, npos)]
            max_fmeasure = max(fmeasures)
            print "Max f_measure for user", userid, ":", max_fmeasure
        else:
            user_fmeasures.append(f_measure(tps, fps, npos))
            print "F_measure for user", userid, ":", f_measure(tps, fps, npos)

    # Compute an average F-measure
    mean_fmeasure = np.mean(user_fmeasures)
    median_fmeasure = np.median(user_fmeasures)
    print "Mean human f_measure:", mean_fmeasure
    print "Median human f_measure:", median_fmeasure

    # Plot the F-measure distribution
    plot_distribution(user_fmeasures,
                      "User F-measure distribution",
                      filename="user_fmeasures.png")

    # human_annotations = parse_csv_annotations(HUMAN_ANNOTATION_PATH)
    # poselets_annotations = parse_matlab_annotations(POSELETS_ANNOTATION_PATH,
    #                                                 min_conf=CONFIDENCE_THRESH)
    # annotations = annotation_list_to_dict(human_annotations
    #                                       + poselets_annotations)
    # scores = {}
    # scores_nocomp = {}
    # fps = {}
    # fns = {}
    # if os.path.exists(MEDIAN_PATH):
    #     os.unlink(MEDIAN_PATH)
    # for imgid, annotation_list in annotations.iteritems():
    #     by_humanity = annotations_by_humanity(annotation_list)
    #     if not by_humanity[True] or not by_humanity[False]:
    #         print imgid, by_humanity[True] == [], by_humanity[False] == []
    #         continue

    #     # cluster the human annotations, and save their medians to disk
    #     clustered_annotations = cluster_annotations(by_humanity[True])
    #     cluster_medians = save_medians(clustered_annotations)

    #     # Assign computer annotations to the closest clusters and add them to the mix.
    #     computer_clusters = assign_clusters(by_humanity[False], cluster_medians)
    #     clustered_annotations += computer_clusters

    #     if random.random() < PCT_IMAGES_TO_PLOT / 100.0:
    #         plotted = plot_annotations(imgid, clustered_annotations)

    #     # compute pairwise error scores for each figure, with and without
    #     # including computer annotations
    #     by_cluster = annotations_by_cluster(clustered_annotations)
    #     all_users = annotations_by_user(clustered_annotations).keys()
    #     overlap_scores = {}
    #     overlap_scores_nocomp = {}
    #     false_positives = {}
    #     false_negatives = {}
    #     for user1, user2 in combinations(all_users, 2):
    #         with_comp = COMP_USER_ID in (user1, user2)
    #         dict_key = str(sorted((user1, user2))) # allow bidirectional lookup
    #         u1_key = str(user1)
    #         u2_key = str(user2)
    #         if not with_comp:
    #             overlap_scores_nocomp[dict_key] = []
    #         overlap_scores[dict_key] = []
    #         false_positives[u1_key] = 0
    #         false_positives[u2_key] = 0
    #         false_negatives[u1_key] = []
    #         false_negatives[u2_key] = []

    #         for cluster_id, annotation_list in by_cluster.iteritems():
    #             u1_false_positives = u2_false_positives = 0
    #             u1_false_negatives = u2_false_negatives = 0
    #             user_annotations = annotations_by_user(annotation_list,
    #                                                    [user1, user2])

    #             # Handle the outlier cluster--these are all false positives.
    #             if cluster_id == -1:
    #                 u1_false_positives += len(user_annotations[user1])
    #                 u2_false_positives += len(user_annotations[user2])
    #                 false_positives[u1_key] = u1_false_positives
    #                 false_positives[u2_key] = u2_false_positives
    #                 continue

    #             # Handle False negatives: users didn't draw a box for this
    #             # figure.
    #             overlap_possible = True
    #             if len(user_annotations[user1]) == 0:
    #                 u1_false_negatives += 1
    #                 overlap_possible = False
    #             if len(user_annotations[user2]) == 0:
    #                 u2_false_negatives += 1
    #                 overlap_possible = False
    #             false_negatives[u1_key].append(u1_false_negatives)
    #             false_negatives[u2_key].append(u2_false_negatives)
    #             if not overlap_possible:
    #                 continue

    #             # Otherwise, overlap scores
    #             if len(user_annotations[user1]) != 1:
    #                 # Either user accidentally drew too many rectangles or
    #                 # algorithm misclassified a rectangle because it was too
    #                 # close to the cluster. Either way, pick a random rectangle
    #                 # to evaluate.
    #                 user_annotations[user1][0] = random.choice(
    #                     user_annotations[user1])
    #             if len(user_annotations[user2]) != 1:
    #                 user_annotations[user2][0] = random.choice(
    #                     user_annotations[user2])
    #             overlap_score = overlap_error_score(
    #                 user_annotations[user1][0][0],
    #                 user_annotations[user2][0][0])
    #             overlap_scores[dict_key].append(overlap_score)
    #             if not with_comp:
    #                 overlap_scores_nocomp[dict_key].append(overlap_score)

    #         # Aggregate False Positives and False Negatives over the image.
    #         # False Positives are fine as is.
    #         # Score False negatives as #FN / #Clusters.
    #         u1_fn_scores = false_negatives[u1_key]
    #         u2_fn_scores = false_negatives[u2_key]
    #         false_negatives[u1_key] = (np.mean(u1_fn_scores)
    #                                    if u1_fn_scores else 0.0)
    #         false_negatives[u2_key] = (np.mean(u2_fn_scores)
    #                                    if u2_fn_scores else 0.0)

    #     # Average overlap scores per user pair per image
    #     scores[imgid] = { key : np.mean(overlap_score_list)
    #                       for key, overlap_score_list
    #                       in overlap_scores.iteritems()
    #                       if overlap_score_list }
    #     scores_nocomp[imgid] = { key : np.mean(overlap_score_list)
    #                              for key, overlap_score_list
    #                              in overlap_scores_nocomp.iteritems()
    #                              if overlap_score_list }
    #     fps[imgid] = false_positives
    #     fns[imgid] = false_negatives

    # # plot the distribution of error between pairs of users on the same image
    # plt.figure()
    # data_same = [error_score for img_dict in scores.itervalues()
    #              for error_score in img_dict.itervalues()]
    # n, bins, patches = plt.hist(data_same, 20, histtype='bar', normed=True)
    # plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
    # plt.xlim([0,1])
    # plt.title("Overlap Error scores (same images)")
    # plt.savefig(os.path.join(RESULTS_PATH, 'overlap_error_scores.png'))
    # plt.show()

    # plt.figure()
    # data_same_nocomp = [error_score for img_dict in scores_nocomp.itervalues()
    #                     for error_score in img_dict.itervalues()]
    # n, bins, patches = plt.hist(data_same_nocomp, 20, histtype='bar', normed=True)
    # plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
    # plt.xlim([0,1])
    # plt.title("Overlap Error scores (same images) (no computer annotations)")
    # plt.savefig(os.path.join(RESULTS_PATH, 'overlap_error_scores_nocomp.png'))
    # plt.show()

    # # Draw a boxplot by image
    # # use ~1/10 of the images so plot is readable
    # boxplot_data = [img_dict.values() for img_dict in scores.values()]
    # num_images = len(boxplot_data)
    # sample_indexes = sorted(random.sample(range(num_images), num_images / 10))
    # boxplot_data = [img_scores for i, img_scores in enumerate(boxplot_data)
    #                 if i in sample_indexes]
    # img_labels = ["Image %s" % str(i) for i in sample_indexes]
    # fig = plt.figure()
    # ax = plt.gca()
    # plt.boxplot(boxplot_data, notch=False, sym='')
    # xtickNames = plt.setp(ax, xticklabels=img_labels)
    # plt.setp(xtickNames, rotation=45, fontsize=8)
    # plt.savefig(os.path.join(RESULTS_PATH, 'overlap_error_scores_box.png'))
    # plt.show()

    # # plot false positives
    # plt.figure()
    # data_fp = [img_dict.values() for i, img_dict in enumerate(fps.values())
    #            if i in sample_indexes]
    # ax = plt.gca()
    # plt.boxplot(data_fp, sym='')
    # xtickNames = plt.setp(ax, xticklabels=img_labels)
    # plt.setp(xtickNames, rotation=45, fontsize=8)
    # plt.title("False Positive Scores")
    # plt.ylim([-1, max([max(scores) for scores in data_fp]) + 2])
    # plt.savefig(os.path.join(RESULTS_PATH, 'false_positives.png'))
    # plt.show()

    # # plot false negatives
    # plt.figure()
    # data_fn = [img_dict.values() for i, img_dict in enumerate(fns.values())
    #            if i in sample_indexes]
    # ax = plt.gca()
    # plt.boxplot(data_fn, sym='')
    # xtickNames = plt.setp(ax, xticklabels=img_labels)
    # plt.setp(xtickNames, rotation=45, fontsize=8)
    # plt.title("False Negative Scores")
    # plt.ylim([-1, 2])
    # plt.savefig(os.path.join(RESULTS_PATH, 'false_negatives.png'))
    # plt.show()

    exit() # Don't wait forever for the last bit!

    # plot the distribution of error scores between pairs of users on different
    # images.
    all_annotations = [annotation for user_annotation_list
                       in annotations.itervalues()
                       for annotation in user_annotation_list]
    data_different = []
    for anno1, anno2 in combinations(all_annotations, 2):
        if anno1.imgid != anno2.imgid:
            data_different.append(overlap_error_score(anno1, anno2))
    plt.figure()
    n, bins, patches = plt.hist(data_different, 20, histtype='bar', normed=True)
    plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
    plt.xlim([0,1])
    plt.title("Overlap Error scores (different images)")
    plt.savefig(os.path.join(RESULTS_PATH, 'overlap_scores_diff_images.png'))
    plt.show()
