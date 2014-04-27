import os

# File paths for saving and loading data
HUMAN_ANNOTATION_PATH = '../annotations/user_annotations.csv'
POSELETS_ANNOTATION_PATH = '../annotations/poselets_annotations.mat'
DPM_ANNOTATION_PATH = '../annotations/dpm_annotations.mat'
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
POSELETS_USERID = 'POSELETS'
DPM_USERID = 'DPM'
COMP_USERIDS = [POSELETS_USERID, DPM_USERID]
