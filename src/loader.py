import csv
from collections import defaultdict
from scipy.io import loadmat

from annotation import Annotation

def annotation_list_to_dict(annotation_list):
    annotations = defaultdict(list)
    for annotation in annotation_list:
        annotations[annotation.imgid].append(annotation)
    return annotations

def parse_csv_annotations(annotation_file, is_human=True):
    annotations = []
    with open(annotation_file, 'rb') as anno_file:
        reader = csv.DictReader(anno_file, delimiter=',')
        for row in reader:
            annotation = Annotation(is_human, raw_row=row)
            annotations.append(annotation)
    return annotations

def parse_matlab_annotations(annotation_file, userid, is_human=False,
                             min_conf=0.0):
    mat = loadmat(annotation_file)
    num_images = len(mat['ids'])
    annotations = []
    for i in range(num_images):
        img_id = mat['ids'][i][0][0] # weird parsing of matlab cells
        for det in mat['ds'][0][i]:
            if not det.any():
                continue
            left, top, width, height, score = det
            if score <= min_conf:
                continue
            annotation = Annotation(
                is_human, confidence=score,
                parsed_attrs={
                    'imgid': img_id,
                    'userid': userid,
                    'left': left,
                    'top': top,
                    'width': width,
                    'height': height
                })
            annotations.append(annotation)
    confidences = sorted(set([anno.confidence for anno in annotations]),
                         reverse=True)
    return annotations, confidences
