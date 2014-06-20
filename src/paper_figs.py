import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from config import DATA_PATH, PAPER_FIGS_PATH

def plot_human_fmeasures(filename="accuracy/human/human_fmeasures.png",
                         show=False):
    data_filename = "user_fmeasure_dist_no bucketing_1.txt"
    with open(os.path.join(DATA_PATH, data_filename), 'rb') as dataf:
        data = [row for row in csv.reader(dataf)]
    x, y = zip(*data)
    fig = plt.figure()
    plt.plot(x, y, 'b-', linewidth=2)
    #fig.axes[0].get_yaxis().set_ticks([])
    plt.title("Distribution of Human Annotator F-Measures")
    plt.xlabel("F-measure score")
    plt.ylabel("Normalized frequency")
    plt.ylim([0, 15])
    plt.xlim([0, 1])
    if filename:
        plt.savefig(os.path.join(PAPER_FIGS_PATH, filename))
    if show:
        plt.show()

def plot_human_fmeasures_buckets(
        filename="degradation/human/fmeasures_buckets.png",
        show=False):
    data_files = ["user_fmeasure_dist_by difficulty_1.txt",
                  "user_fmeasure_dist_by difficulty_2.txt",
                  "user_fmeasure_dist_by difficulty_3.txt",
                  "user_fmeasure_dist_by difficulty_4.txt",
                  "user_fmeasure_dist_by difficulty_5.txt"]
    styles = ['b-',
              'r-',
              'g-',
              'y-',
              'm-']
    labels = ['Bucket %d' % i for i in range(1,6)]
    fig = plt.figure()
    for data_file, style, label in zip(data_files, styles, labels):
        with open(os.path.join(DATA_PATH, data_file), 'rb') as dataf:
            data = [row for row in csv.reader(dataf)]
        x, y = zip(*data)
        plt.plot(x, y, style, label=label, linewidth=2)
    plt.title("F-measure Distributions with Varying Image Difficulty")
    plt.xlabel("F-measure score")
    plt.ylabel("Normalized frequency")
    plt.ylim([0, 15])
    plt.xlim([0, 1])
    plt.legend(loc='upper left')

    if filename:
        plt.savefig(os.path.join(PAPER_FIGS_PATH, filename))
    if show:
        plt.show()

def plot_robot_pr(filename="accuracy/robot/prec_recall.png",
                  show=False):
    data_files = ["DPM_prec_recall_No bucketing_1.txt",
                  "POSELETS_prec_recall_No bucketing_1.txt",
                  "RCNN_prec_recall_No bucketing_1.txt",
                  "D&T_prec_recall_No bucketing_1.txt"]
    styles = ['b-',
              'r-',
              'g-',
              'y-']
    labels = ['DPM',
              'Poselets',
              'RCNN',
              'Dalal & Triggs']
    fig = plt.figure()
    for data_file, style, label in zip(data_files, styles, labels):
        with open(os.path.join(DATA_PATH, data_file), 'rb') as dataf:
            data = [row for row in csv.reader(dataf)]
        x, y = zip(*data)
        plt.plot(x, y, style, label=label)
    plt.title("Precision-Recall curves by algorithm")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend(loc='upper right')
    if filename:
        plt.savefig(os.path.join(PAPER_FIGS_PATH, filename))
    if show:
        plt.show()

def plot_cmp_pr(filename="accuracy/both/prec_recall.png",
                show=False):
    data_files = ["DPM_prec_recall_No bucketing_1.txt",
                  "POSELETS_prec_recall_No bucketing_1.txt",
                  "RCNN_prec_recall_No bucketing_1.txt",
                  "D&T_prec_recall_No bucketing_1.txt"]
    styles = ['b-',
              'r-',
              'g-',
              'y-']
    labels = ['DPM',
              'Poselets',
              'RCNN',
              'Dalal & Triggs']
    mean_human_prec = 0.807190829731
    mean_human_recall = 0.86249085525
    median_human_prec = 0.810344827586
    median_human_recall = 0.88785046729
    fig = plt.figure()
    for data_file, style, label in zip(data_files, styles, labels):
        with open(os.path.join(DATA_PATH, data_file), 'rb') as dataf:
            data = [row for row in csv.reader(dataf)]
        x, y = zip(*data)
        plt.plot(x, y, style, label=label)
    plt.plot([mean_human_prec], [mean_human_recall], 'co', label='Mean Human')
    plt.plot([median_human_prec], [median_human_recall], 'ko',
             label='Median Human')
    plt.title("Precision-Recall curves by algorithm.")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    legend = plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    if filename:
        plt.savefig(os.path.join(PAPER_FIGS_PATH, filename),
                    bbox_extra_artists=(legend,), bbox_inches='tight')
    if show:
        plt.show()

def plot_robot_pr_buckets(filename="degradation/robot/prec_recall_buckets.png",
                          show=False):
    data_files = ["DPM_prec_recall_By Difficulty_1.txt",
                  "DPM_prec_recall_By Difficulty_2.txt",
                  "DPM_prec_recall_By Difficulty_3.txt",
                  "DPM_prec_recall_By Difficulty_4.txt",
                  "DPM_prec_recall_By Difficulty_5.txt"]
    styles = ['b-',
              'r-',
              'g-',
              'y-',
              'm-']
    labels = ['Bucket %d' % i for i in range(1,6)]
    fig = plt.figure()
    for data_file, style, label in zip(data_files, styles, labels):
        with open(os.path.join(DATA_PATH, data_file), 'rb') as dataf:
            data = [row for row in csv.reader(dataf)]
        x, y = zip(*data)
        plt.plot(x, y, style, label=label)
    plt.title("Precision-Recall curves by algorithm")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend(loc='upper right')
    if filename:
        plt.savefig(os.path.join(PAPER_FIGS_PATH, filename))
    if show:
        plt.show()

def plot_cmp_fmeasures_buckets(
        filename='degradation/both/fmeasures_buckets.png',
        show=False):
    fig = plt.figure()
    ax = plt.axes()
    X = np.arange(5)
    bar_width = 0.15

    median_human_fmeasures = [
        1.0,
        1.0,
        0.900763358779,
        0.888888888889,
        0.755244755245,
    ]
    medianhf_rects = ax.bar(X, median_human_fmeasures, bar_width,
                            color='k', label='Median Human')

    mean_human_fmeasures = [
        0.969222689076,
        0.933414354953,
        0.871949046235,
        0.846665087375,
        0.754370584503,
    ]
    meanhf_rects = ax.bar(X + bar_width, mean_human_fmeasures, bar_width, color='c',
                          label='Mean Human')

    DPM_max_fmeasures = [
        0.666666666667,
        0.736842105263,
        0.534031413613,
        0.453038674033,
        0.368131868132,
    ]
    DPM_rects = ax.bar(X + 2*bar_width, DPM_max_fmeasures, bar_width, color='b',
                       label='DPM')

    POSELETS_max_fmeasures = [
        0.555555555556,
        0.634146341463,
        0.425287356322,
        0.196581196581,
        0.108045977011,
    ]
    POSELETS_rects = ax.bar(X + 3*bar_width, POSELETS_max_fmeasures, bar_width,
                            color='r', label='Poselets')

    RCNN_max_fmeasures = [
        0.6,
        0.387096774194,
        0.191176470588,
        0.236220472441,
        0.189189189189,
    ]
    RCNN_rects = ax.bar(X + 4*bar_width, RCNN_max_fmeasures, bar_width,
                        color='g', label='RCNN')

    DT_max_fmeasures = [
        0.285714285714,
        0.0689655172414,
        0.0673076923077,
        0.0549635445878,
        0.0451875282422
    ]
    DT_rects = ax.bar(X + 5*bar_width, DT_max_fmeasures, bar_width, color='y',
                      label='Dalal & Triggs')

    plt.title('Human and algorithm F-measures by bucket.')
    plt.ylabel('F-measure score')
    ax.set_xticks(X + 3*bar_width)
    ax.set_xticklabels(('Bucket 1', 'Bucket 2', 'Bucket 3', 'Bucket 4',
                        'Bucket 5'))
    legend = ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

    if filename:
        plt.savefig(os.path.join(PAPER_FIGS_PATH, filename),
                    bbox_extra_artists=(legend,), bbox_inches='tight')
    if show:
        plt.show()

def plot_dpm_time_buckets(filename="methods/time_buckets.png",
                          show=False):
    data_files = ["DPM_prec_recall_By Time (Equidepth)_1.txt",
                  "DPM_prec_recall_By Time (Equidepth)_2.txt",
                  "DPM_prec_recall_By Time (Equidepth)_3.txt",
                  "DPM_prec_recall_By Time (Equidepth)_4.txt",
                  "DPM_prec_recall_By Time (Equidepth)_5.txt"]
    styles = ['b-',
              'r-',
              'g-',
              'y-',
              'm-']
    labels = ['Bucket %d' % i for i in range(1,6)]
    fig = plt.figure()
    for data_file, style, label in zip(data_files, styles, labels):
        with open(os.path.join(DATA_PATH, data_file), 'rb') as dataf:
            data = [row for row in csv.reader(dataf)]
        x, y = zip(*data)
        plt.plot(x, y, style, label=label)
    plt.title("DPM Precision-Recall curves bucketed by time.")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend(loc='upper right')
    if filename:
        plt.savefig(os.path.join(PAPER_FIGS_PATH, filename))
    if show:
        plt.show()

if __name__ == '__main__':
    plot_human_fmeasures()
    plot_human_fmeasures_buckets()
    plot_robot_pr()
    plot_cmp_pr()
    plot_robot_pr_buckets()
    plot_cmp_fmeasures_buckets()
    plot_dpm_time_buckets()
