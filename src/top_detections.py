import csv
from annotation import annotations_by_confidence
from buckets import assign_buckets
from config import ALGORITHMS, CONFIDENCE_THRESH, HUMAN_ANNOTATION_PATH
from eval import eval_user
from loader import parse_csv_annotations, load_algorithm, annotation_list_to_dict

def top_detections_by_bucket(bucket_name, n):
    bucket = [b for b in assign_buckets()[0] if b.name == bucket_name][0]
    algorithms = {}
    human_annotations = annotation_list_to_dict(
        parse_csv_annotations(HUMAN_ANNOTATION_PATH))

    for algorithm in ALGORITHMS:
        top_dets = []
        algo_name = algorithm['name']
        load_algorithm(algorithm, CONFIDENCE_THRESH, algorithms)
        annos = algorithms[algo_name][0]

        for imgid in bucket.imgids:
            gt_annos = human_annotations[imgid]
            comp_annos = annotations_by_confidence(annos[imgid])[0:n]
            tps, _, _ = eval_user(comp_annos, gt_annos)
            for i, tp in enumerate(tps):
                comp_annos[i].is_tp = tp == 1
                top_dets.append(comp_annos[i])
        outputf = 'top_det_%s.csv' % algo_name
        data = [(anno.confidence, anno.imgid, bucket_name, anno.left, anno.top,
                 anno.width, anno.height, anno.is_tp)
                for anno in annotations_by_confidence(top_dets)]
        with open(outputf, 'wb') as outf:
            csv.writer(outf).writerows(data[0:n])

if __name__ == '__main__':
    top_detections_by_bucket(2, 10)
