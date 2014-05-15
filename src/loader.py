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

def load_algorithm(algorithm_config, min_conf, append_to):
    annotations, confidences = parse_matlab_annotations(
        algorithm_config['annotation_path'],
        algorithm_config['name'],
        min_conf=min_conf)
    append_to[algorithm_config['name']] = (annotation_list_to_dict(annotations), 
                                           confidences)

def parse_matlab_annotations(annotation_file, userid, is_human=False,
                             min_conf=0.0):
    mat = loadmat(annotation_file)
    if mat.has_key('ids'):
        img_id_key = 'ids'
    elif mat.has_key('image_ids'):
        img_id_key = 'image_ids'
    else:
        raise ValueError(".mat file contains no list of image ids.")

    if mat.has_key('ds'):
        bbox_key = 'ds'
    elif mat.has_key('region_proposals'):
        bbox_key = 'region_proposals'
    else:
        raise ValueError(".mat file contains no list of detections.")

    num_images = len(mat[img_id_key])
    annotations = []
    for i in range(num_images):
        img_id = mat[img_id_key][i][0][0] # weird parsing of matlab cells
        for det in mat[bbox_key][0][i]:
            if not det.any():
                continue
#            left, top, width, height, score = det
            left, top, right, bottom, score = det
            if score <= min_conf:
                continue
            annotation = Annotation(
                is_human, confidence=score,
                parsed_attrs={
                    'imgid': img_id,
                    'userid': userid,
                    'left': left,
                    'top': top,
                    'width': right - left + 1,
                    'height': bottom - top + 1,
                })
            annotations.append(annotation)
    confidences = sorted(set([anno.confidence for anno in annotations]),
                         reverse=True)
    return annotations, confidences
