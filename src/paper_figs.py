import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
from buckets import Bucket, assign_buckets
from config import DATA_PATH, PAPER_FIGS_PATH

COLORS = {
    'POSELETS': '#1b9e77',
    'DPM': '#d95f02',
    'RCNN': '#7570b3',
    'D&T': '#e7298a',
    'HUMAN': '#66a61e',

    'B1': '#1f78b4',
    'B2': '#b2df8a',
    'B3': '#a6cee3',
    'B4': '#33a02c',
    'B5': '#fb9a99',
    'NEUTRAL': '#1e1e1e',
}

def plot_human_fmeasures(filename="accuracy/human/human_fmeasures.png",
                         show=False):
    data_filename = "user_fmeasure_dist_no bucketing_1.txt"
    with open(os.path.join(DATA_PATH, data_filename), 'rb') as dataf:
        data = [row for row in csv.reader(dataf)]
    x, y = zip(*data)
    y = np.array([float(n) for n in y])
    y = y / max(y)
    fig = plt.figure()
    plt.plot(x, y, '-', color=COLORS['HUMAN'], linewidth=2)
    plt.xlabel("F-measure")
    plt.ylabel("Normalized frequency")
    plt.ylim([0, 1.1])
    plt.xlim([0, 1])

    font = {'size': 26}
    plt.rc('font', **font)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
    plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower', nbins=5))
    plt.gca().grid(True)

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
    styles = ['B1',
              'B2',
              'B3',
              'B4',
              'B5']
    labels = ['Bucket %d' % i for i in range(1,6)]
    fig = plt.figure()
    max_val = 0
    xs = []
    ys = []
    for data_file in data_files:
        with open(os.path.join(DATA_PATH, data_file), 'rb') as dataf:
            data = [row for row in csv.reader(dataf)]
        x, y = zip(*data)
        y = np.array([float(n) for n in y])
        xs.append(x)
        ys.append(y)
        if max(y) > max_val:
            max_val = max(y)

    for x, y, style, label in zip(xs, ys, styles, labels):
        y = y / max_val
        plt.plot(x, y, '-', color=COLORS[style], label=label, linewidth=2)

    plt.xlabel("F-measure")
    plt.ylabel("Normalized frequency")
    plt.ylim([0, 1.1])
    plt.xlim([0, 1])
    plt.legend(loc='upper left', labelspacing=0.1, handletextpad=0.05,
               handlelength=1.5)

    font = {'size': 14}
    plt.rc('font', **font)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
    plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower', nbins=5))
    plt.gca().grid(True)

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
    styles = ['DPM',
              'POSELETS',
              'RCNN',
              'D&T']
    labels = ['DPM',
              'Poselets',
              'RCNN',
              'D&T']
    aps = [0.378, 0.178, 0.104, 0.0193]
    max_prs = [(0.463722397476, 0.444108761329, 0.458),
               (0.239747634069, 0.311475409836, 0.271),
               (0.17665615142, 0.314606741573, 0.226),
               (0.485804416404, 0.0271365638767, 0.051),]
    fig = plt.figure()
    for data_file, style, label, ap in zip(data_files, styles, labels, aps):
        with open(os.path.join(DATA_PATH, data_file), 'rb') as dataf:
            data = [row for row in csv.reader(dataf)]
        x, y = zip(*data)
        plt.plot(x, y, '-', color=COLORS[style], label=label + ': AP(' + str(ap) + ')')

    for style, (recall, precision, f_measure) in zip(styles, max_prs):
        plt.plot(recall, precision, 'o', color=COLORS[style],
                 label='F1(' + str(f_measure) + ')')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    legend = plt.legend(loc='upper right', prop={'size':14}, numpoints=1,
                        ncol=2, columnspacing=0.5, handlelength=0.8,
                        handletextpad=0.2)
    font = {'size': 26}
    plt.rc('font', **font)
    plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower', nbins=5))
    plt.gca().grid(True)

    if filename:
        plt.savefig(os.path.join(PAPER_FIGS_PATH, filename),
                    bbox_extra_artists=(legend,), bbox_inches='tight')
    if show:
        plt.show()

def plot_cmp_pr(filename="accuracy/both/prec_recall_legend_outside.png",
                show=False):
    data_files = ["DPM_prec_recall_No bucketing_1.txt",
                  "POSELETS_prec_recall_No bucketing_1.txt",
                  "RCNN_prec_recall_No bucketing_1.txt",
                  "D&T_prec_recall_No bucketing_1.txt"]
    styles = ['DPM',
              'POSELETS',
              'RCNN',
              'D&T']
    labels = ['DPM',
              'Poselets',
              'RCNN',
              'D&T']
    aps = [0.378, 0.178, 0.104, 0.0193]
    max_prs = [(0.463722397476, 0.444108761329, 0.458),
               (0.239747634069, 0.311475409836, 0.271),
               (0.17665615142, 0.314606741573, 0.226),
               (0.485804416404, 0.0271365638767, 0.051),]
    mean_human_prec = 0.807190829731
    mean_human_recall = 0.86249085525
    mean_human_fmeasure = 0.829
    median_human_prec = 0.810344827586
    median_human_recall = 0.88785046729
    fig = plt.figure()
    for data_file, style, label, (recall, precision, f_measure), ap in zip(data_files, styles, labels, max_prs, aps):
        with open(os.path.join(DATA_PATH, data_file), 'rb') as dataf:
            data = [row for row in csv.reader(dataf)]
        x, y = zip(*data)
        plt.plot(x, y, '-', color=COLORS[style],
                 label=label + ': AP(' + str(ap) + ')')
    plt.plot([mean_human_prec], [mean_human_recall], 'o', color=COLORS['HUMAN'],
             label='Humans: F1(' + str(mean_human_fmeasure) + ')')

    for style, (recall, precision, f_measure) in zip(styles, max_prs):
        plt.plot(recall, precision, 'o', color=COLORS[style],
                 label='F1(' + str(f_measure) + ')')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
#    legend = plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,
#                        prop={'size':14}, numpoints=1)

    legend = plt.legend(bbox_to_anchor=(1.01, 1), loc=2, prop={'size':14}, numpoints=1,
                        ncol=2, columnspacing=0.5, handlelength=0.8, borderaxespad=0.,
                        handletextpad=0.2)
#    legend = plt.legend(bbox_to_anchor=(0.41, 0.845), loc=2, prop={'size':14}, numpoints=1,
#                        ncol=2, columnspacing=0.5, handlelength=0.8,
#                        handletextpad=0.2)


    font = {'size': 26}
    plt.rc('font', **font)
    plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower', nbins=5))
    plt.gca().grid(True)

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
    styles = ['B1',
              'B2',
              'B3',
              'B4',
              'B5']
    labels = ['Bucket %d' % i for i in range(1,6)]
    aps = [0.442, 0.701, 0.435, 0.303, 0.273]
    max_prs = [(0.6, 0.75, 0.667),
               (0.666666666667, 0.823529411765, 0.737),
               (0.566666666667, 0.50495049505, 0.534),
               (0.450549450549, 0.455555555556, 0.453),
               (0.638095238095, 0.258687258687, 0.368)]

    fig = plt.figure()
    for data_file, style, label, (recall, precision, f_measure), ap in zip(data_files, styles, labels, max_prs, aps):
        with open(os.path.join(DATA_PATH, data_file), 'rb') as dataf:
            data = [row for row in csv.reader(dataf)]
        x, y = zip(*data)
        plt.plot(x, y, '-', color=COLORS[style],
                 label=label + ': AP(' + str(ap) + ')')

    for style, (recall, precision, f_measure) in zip(styles, max_prs):
        plt.plot(recall, precision, 'o', color=COLORS[style],
                 label='F1(' + str(f_measure) + ')')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    #    legend = plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,
    #                        prop={'size':14}, numpoints=1)
    legend = plt.legend(bbox_to_anchor=(0.02, 0.005), loc=3, numpoints=1,
                        ncol=2, borderaxespad=0., columnspacing=0.1,
                        prop={'size':14}, labelspacing=0.1, handletextpad=0.1,
                        handlelength=1.0)

    font = {'size': 26}
    plt.rc('font', **font)
    plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower', nbins=5))
    plt.gca().grid(True)

    if filename:
        plt.savefig(os.path.join(PAPER_FIGS_PATH, filename),
                   bbox_extra_artists=(legend,), bbox_inches='tight')
    if show:
        plt.show()

# def plot_cmp_pr_buckets(filename="degradation/both/prec_recall_buckets.png",
#                           show=False):
#     data_files = ["DPM_prec_recall_By Difficulty_1.txt",
#                   "DPM_prec_recall_By Difficulty_2.txt",
#                   "DPM_prec_recall_By Difficulty_3.txt",
#                   "DPM_prec_recall_By Difficulty_4.txt",
#                   "DPM_prec_recall_By Difficulty_5.txt"]
#     styles = ['b-',
#               'r-',
#               'k-',
#               'c-',
#               'm-']
#     labels = ['Bucket %d' % i for i in range(1,6)]
#     aps = [0.442, 0.701, 0.435, 0.303, 0.273]
#     max_prs = [(0.6, 0.75, 0.667),
#                (0.666666666667, 0.823529411765, 0.737),
#                (0.566666666667, 0.50495049505, 0.534),
#                (0.450549450549, 0.455555555556, 0.453),
#                (0.638095238095, 0.258687258687, 0.368)]
#     human_prs = [(0.9921875, 0.950047348485, 0.969),
#                  (0.943374272786, 0.925371581513, 0.933),
#                  (0.891848941784, 0.851475586862, 0.869),
#                  (0.874703199308, 0.830823579113, 0.848),
#                  (0.796548165736, 0.723322393159, 0.754)]

#     fig = plt.figure()
#     for (data_file, style, label, (recall, precision, f_measure), ap,
#          (h_recall, h_precision, h_fmeasure)) in zip(data_files, styles, labels, max_prs, aps, human_prs):
#         with open(os.path.join(DATA_PATH, data_file), 'rb') as dataf:
#             data = [row for row in csv.reader(dataf)]
#         x, y = zip(*data)
#         plt.plot(x, y, style, label=label + ': AP=' + str(ap))
#         plt.plot(recall, precision, style[0]+'o',
#                  label='Max F-measure = ' + str(f_measure))
#         plt.plot(h_recall, h_precision, style[0]+'D',
#                  label='Human avg. F-measure = ' + str(h_fmeasure))

#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.ylim([0, 1])
#     plt.xlim([0, 1])
#     legend = plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,
#                         prop={'size':14}, numpoints=1)

#     font = {'size': 26}
#     plt.rc('font', **font)
#     plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower', nbins=5))

#     if filename:
#         plt.savefig(os.path.join(PAPER_FIGS_PATH, filename),
#                    bbox_extra_artists=(legend,), bbox_inches='tight')
#     if show:
#         plt.show()

def plot_cmp_fmeasures_buckets(
        filename='degradation/both/fmeasures_buckets.png',
        show=False):
    fig = plt.figure()
    ax = plt.axes()
    X = np.arange(5)
    bar_width = 0.15

    # median_human_fmeasures = [
    #     1.0,
    #     1.0,
    #     0.900763358779,
    #     0.888888888889,
    #     0.755244755245,
    # ]
    # medianhf_rects = ax.bar(X, median_human_fmeasures, bar_width,
    #                         color='m', label='Median Human')

    mean_human_fmeasures = [
        0.969222689076,
        0.933414354953,
        0.871949046235,
        0.846665087375,
        0.754370584503,
    ]
    meanhf_rects = ax.bar(X + bar_width, mean_human_fmeasures, bar_width,
                          color=COLORS['HUMAN'], label='Human')

    DPM_max_fmeasures = [
        0.666666666667,
        0.736842105263,
        0.534031413613,
        0.453038674033,
        0.368131868132,
    ]
    DPM_rects = ax.bar(X + 2*bar_width, DPM_max_fmeasures, bar_width,
                       color=COLORS['DPM'], label='DPM')

    POSELETS_max_fmeasures = [
        0.555555555556,
        0.634146341463,
        0.425287356322,
        0.196581196581,
        0.108045977011,
    ]
    POSELETS_rects = ax.bar(X + 3*bar_width, POSELETS_max_fmeasures, bar_width,
                            color=COLORS['POSELETS'], label='Poselets')

    RCNN_max_fmeasures = [
        0.6,
        0.387096774194,
        0.191176470588,
        0.236220472441,
        0.189189189189,
    ]
    RCNN_rects = ax.bar(X + 4*bar_width, RCNN_max_fmeasures, bar_width,
                        color=COLORS['RCNN'], label='RCNN')

    DT_max_fmeasures = [
        0.285714285714,
        0.0689655172414,
        0.0673076923077,
        0.0549635445878,
        0.0451875282422
    ]
    DT_rects = ax.bar(X + 5*bar_width, DT_max_fmeasures, bar_width,
                      color=COLORS['D&T'], label='D&T')

    plt.ylabel('F-measure score')
    plt.xlabel('Distortion Bucket')
    ax.set_xticks(X + 3*bar_width)
    ax.set_xticklabels(('1', '2', '3', '4', '5'))

    legend = ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,
                       prop={'size':14})

    legend = plt.legend(bbox_to_anchor=(0.71, 0.82), loc=3, borderaxespad=0.,
                        prop={'size':14}, labelspacing=0.1, handletextpad=0.15,
                        handlelength=0.5, ncol=2, columnspacing=0.05)

    font = {'size': 26}
    plt.rc('font', **font)
    plt.gca().grid(True)

    if filename:
        plt.savefig(os.path.join(PAPER_FIGS_PATH, filename),
                    bbox_extra_artists=(legend,), bbox_inches='tight')
    if show:
        plt.show()

# def plot_dpm_time_buckets(filename="methods/time_buckets.png",
#                           show=False):
#     data_files = ["DPM_prec_recall_By Time (Equidepth)_1.txt",
#                   "DPM_prec_recall_By Time (Equidepth)_2.txt",
#                   "DPM_prec_recall_By Time (Equidepth)_3.txt",
#                   "DPM_prec_recall_By Time (Equidepth)_4.txt",
#                   "DPM_prec_recall_By Time (Equidepth)_5.txt"]
#     styles = ['b-',
#               'r-',
#               'k-',
#               'c-',
#               'm-']
#     labels = ['Bucket %d' % i for i in range(1,6)]
#     fig = plt.figure()
#     for data_file, style, label in zip(data_files, styles, labels):
#         with open(os.path.join(DATA_PATH, data_file), 'rb') as dataf:
#             data = [row for row in csv.reader(dataf)]
#         x, y = zip(*data)
#         plt.plot(x, y, style, label=label)
#     plt.title("DPM Precision-Recall curves bucketed by time.")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.ylim([0, 1])
#     plt.xlim([0, 1])
#     plt.legend(loc='upper right')
#     if filename:
#         plt.savefig(os.path.join(PAPER_FIGS_PATH, filename))
#     if show:
#         plt.show()

def plot_bucket_histogram(filename="methods/buckets_difficulty_histogram.png",
                          show=False):
    difficulty_buckets, _, _, _ = assign_buckets()
    Y = [len(bucket.imgids) for bucket in difficulty_buckets]
    X = np.arange(len(Y)) # the x locations for the groups
    width = 1.0       # the width of the bars

    fig = plt.figure()
    ax = plt.axes()
    rects = ax.bar(X, Y, width, color=COLORS['HUMAN'])

    plt.ylabel('Number of images')
    plt.xlabel('Distortion Bucket')
    ax.set_xticks(X + 0.5 * width)
    ax.set_xticklabels(('1', '2', '3', '4', '5'))
    font = {'size': 26}
    plt.rc('font', **font)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
    plt.gca().grid(True)

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
    #plot_cmp_pr_buckets()
    plot_cmp_fmeasures_buckets()
    # plot_dpm_time_buckets()
    plot_bucket_histogram()
