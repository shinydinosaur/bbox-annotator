import os

# File paths for saving and loading data
ANNOTATION_DIR = '../annotations'
HUMAN_ANNOTATION_PATH = os.path.join(ANNOTATION_DIR, 'user_annotations.csv')
RESULTS_PATH = '../results'
MEDIAN_PATH = os.path.join(RESULTS_PATH, 'cluster_medians.csv')
IMAGE_DIR = '/Users/dhaas/Code/projects/Art_Vision/Picasso/People/'

# Plotting configuration
PCT_IMAGES_TO_PLOT = 5
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

# Evaluation configuration
CONFIDENCE_THRESH = -500.0
OVERLAP_THRESH = 0.5

# Algorithm config
ALGORITHMS = [
    {
        'name': 'POSELETS',
        'annotation_path' : os.path.join(ANNOTATION_DIR,
                                         'poselets_annotations.mat'),
    },
    {
        'name': 'DPM',
        'annotation_path' : os.path.join(ANNOTATION_DIR,
                                         'dpm_annotations.mat'),
    },
 #   {
 #       'name': 'RCNN',
 #       'annotation_path' : os.path.join(ANNOTATION_DIR,
 #                                        'rcnn_annotations.mat'),
 #   },
]
