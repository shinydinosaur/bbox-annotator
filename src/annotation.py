import numpy as np
import os
from operator import attrgetter
from scipy.spatial.distance import pdist

class Annotation(object):
    def __init__(self, is_human, confidence=100,
                 raw_row=None, parsed_attrs=None):
        self.is_human = is_human
        self.confidence = confidence
        if raw_row:
            parsed_attrs = self.parse_raw_row(raw_row)
        self._init_parsed(parsed_attrs)

    def parse_raw_row(self, raw_row):
        return {
            'imgid': os.path.splitext(os.path.basename(raw_row['imgid']))[0],
            'userid': int(raw_row['userid']),
            'left': float(raw_row['bbox_left']),
            'top': float(raw_row['bbox_top']),
            'width': float(raw_row['bbox_width']),
            'height': float(raw_row['bbox_height']),
        }

    def _init_parsed(self, parsed_attrs):
        for prop, val in parsed_attrs.iteritems():
            setattr(self, prop, val)

    def area(self):
        return self.width * self.height

    def centroid(self):
        return ((self.left + self.width) / 2.0,
                (self.top + self.height) / 2.0)

    def feature_vector(self):
        # try clustering on the centroids of the boxes
        # return self.centroid()

        # try clustering on all corners of the boxes
        return np.array((self.left,
                         self.left + self.width,
                         self.top,
                         self.top + self.height))

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
        intersection_right = min((
            self.left + self.width,
            other_annotation.left + other_annotation.width))
        intersection_bottom = min((
            self.top + self.height,
            other_annotation.top + other_annotation.height))
        intersection_width = intersection_right - intersection_left
        intersection_height = intersection_bottom - intersection_top
        return intersection_width * intersection_height

    def union_size(self, other_annotation):
        return (self.area()
                + other_annotation.area()
                - self.intersection_size(other_annotation))

    def distance_to(self, other_annotation):
        # distance defined by distance between centroids.
        # TODO(dhaas): use another metric to handle overlapping clusters better?
        #   Hausdorff distance seems like a good candidate
        return pdist([self.centroid(), other_annotation.centroid()])[0]

def unique_users(annotation_lists):
    return set([annotation.userid for annotations in annotation_lists
                for annotation in annotations])

def split_annotations(annotations, userid):
    return ([anno for anno in annotations if anno.userid == userid],
            [anno for anno in annotations if anno.userid != userid])

def median_annotation(annotations):
    if not annotations:
        return None
    left = np.median([annotation[0].left for annotation in annotations])
    top = np.median([annotation[0].top for annotation in annotations])
    right =  np.median([annotation[0].left + annotation[0].width
                        for annotation in annotations])
    bottom =  np.median([annotation[0].top + annotation[0].height
                         for annotation in annotations])
    return Annotation(True, confidence=1.0,
                      parsed_attrs={
                          'imgid': annotations[0][0].imgid,
                          'userid': -1,
                          'left': left,
                          'top': top,
                          'width': right - left,
                          'height': bottom - top,
                      })

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

def annotations_by_humanity(annotations, clustered=False):
    by_humanity = {True: [], False: []}
    if not clustered:
        for annotation in annotations:
            by_humanity[annotation.is_human].append(annotation)
    else:
        for annotation, cluster_id in annotations:
            by_huamnity[annotation.is_human].append((annotation, cluster_id))
    return by_humanity

def annotations_by_confidence(annotations):
    return sorted(annotations, key=attrgetter('confidence'), reverse=True)
