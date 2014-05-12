import csv
import numpy as np

from config import (BUCKET_INPUT_PATH,
                    DIFFICULTY_BUCKET_OUTPUT_PATH,
                    EQUIWIDTH_TIME_BUCKET_OUTPUT_PATH,
                    EQUIDEPTH_TIME_BUCKET_OUTPUT_PATH)
from plot import plot_distribution

class Bucket(object):
    def __init__(self, images, bucket_method, bucket_id):
        self.imgids = images
        self.method = bucket_method
        self.name = bucket_id

    def histogram_values(self):
        return [self.name]*len(self.imgids)

    @classmethod
    def get_bucket(cls, imgid, buckets):
        for bucket in buckets:
            if imgid in bucket.imgids:
                return bucket.name
        raise ValueError("imgid not in any of the passed buckets.")

    @classmethod
    def serialize_buckets(cls, buckets, output_path):
        out_rows = []
        for bucket in buckets:
            out_rows.extend([(imgid, bucket.name)
                             for imgid in bucket.imgids])
        with open(output_path, 'wb') as csv_file:
            csv.writer(csv_file).writerows(out_rows)

    @classmethod
    def to_histogram_values(cls, buckets):
        return [value
                for bucket in buckets
                for value in bucket.histogram_values()]

    class Methods:
        DIFFICULTY = 0
        TIME_EQUIWIDTH = 1
        TIME_EQUIDEPTH = 2

def load_bucket_rows():
    rows = []
    with open(BUCKET_INPUT_PATH, 'rb') as bucket_csv:
        for row in csv.DictReader(bucket_csv):
            rows.append(row)
    return group_by_image(rows)

def group_by_image(rows):
    images = {}
    for row in rows:
        imgid = row['imgid']
        if imgid not in images:
            images[imgid] = []
        images[imgid].append(row)
    return images

def assign_bucket_by_difficulty(img_difficulty_scores):
    """group images into buckets by median user-rated difficulty.
    Buckets are numbered 1-5 for difficulty rating. This is an equiwidth
    bucketing, not an equidepth one."""
    return round(np.median(img_difficulty_scores))

def assign_bucket_by_time(img_median_time, all_median_times, equiwidth=True):
    """group images into buckets by the median user time-to-completion.
    5 buckets are created. If equiwidth, buckets are evenly spaced along
    the range of completion times. If equidepth, each bucket contains 20%
    of the images ordered by median time-to-completion."
    """
    max_time = max(all_median_times)
    min_time = min(all_median_times)
    print min_time, max_time
    if equiwidth:
        bucket_size = (max_time - min_time) / 5.0
        bucket_ends = [min_time + bucket_size,
                       min_time + 2.0*bucket_size,
                       min_time + 3.0*bucket_size,
                       min_time + 4.0*bucket_size,
                       min_time + 5.0*bucket_size]
    else:
        bucket_ends = np.percentile(all_median_times, (20, 40, 60, 80, 100))
    for bucket_index, bucket_end in enumerate(bucket_ends):
        print bucket_index, bucket_end
        if img_median_time <= bucket_end:
            return bucket_index + 1 # get an id in [1, 5]
    raise ValueError("img_median_time exceeds maximum median time")

def get_median_times_by_image(rows_by_image):
    return { imgid : np.median([float(img_row['time']) for img_row in img_rows])
             for imgid, img_rows in rows_by_image.iteritems() }

def assign_buckets(serialize=False, plot=False):
    data = load_bucket_rows()
    median_times_by_image = get_median_times_by_image(data)
    all_median_times = [median_time
                        for median_time in median_times_by_image.itervalues()]
    buckets_by_difficulty = {1:[], 2:[], 3:[], 4:[], 5:[]}
    buckets_by_time_equiwidth = {1:[], 2:[], 3:[], 4:[], 5:[]}
    buckets_by_time_equidepth = {1:[], 2:[], 3:[], 4:[], 5:[]}
    for imgid, rows in data.iteritems():
        buckets_by_difficulty[
            assign_bucket_by_difficulty(
                [int(row['difficulty']) for row in rows])].append(imgid)

        img_median_time = median_times_by_image[imgid]
        buckets_by_time_equiwidth[
            assign_bucket_by_time(
                img_median_time, all_median_times)].append(imgid)

        buckets_by_time_equidepth[
            assign_bucket_by_time(
                img_median_time, all_median_times,
                equiwidth=False)].append(imgid)

    difficulty_buckets = []
    equiwidth_time_buckets = []
    equidepth_time_buckets = []
    for bucket_id in range(1, 6):
        difficulty_buckets.append(
            Bucket(buckets_by_difficulty[bucket_id],
                   Bucket.Methods.DIFFICULTY,
                   bucket_id))

        equiwidth_time_buckets.append(
            Bucket(buckets_by_time_equiwidth[bucket_id],
                   Bucket.Methods.TIME_EQUIWIDTH,
                   bucket_id))

        equidepth_time_buckets.append(
            Bucket(buckets_by_time_equidepth[bucket_id],
                   Bucket.Methods.TIME_EQUIDEPTH,
                   bucket_id))
    if serialize:
        Bucket.serialize_buckets(difficulty_buckets,
                                 DIFFICULTY_BUCKET_OUTPUT_PATH)
        Bucket.serialize_buckets(equiwidth_time_buckets,
                                 EQUIWIDTH_TIME_BUCKET_OUTPUT_PATH)
        Bucket.serialize_buckets(equidepth_time_buckets,
                                 EQUIDEPTH_TIME_BUCKET_OUTPUT_PATH)

    if plot:
        plot_distribution(
            Bucket.to_histogram_values(difficulty_buckets),
            "Number of images per bucket (bucketed by difficulty)",
            nbins=5,
            set_lims=False,
            normed=False,
            filename='buckets_difficulty_histogram.png')
        plot_distribution(
            Bucket.to_histogram_values(equiwidth_time_buckets),
            "Number of images per bucket (bucketed by Time: equiwidth)",
            nbins=5,
            set_lims=False,
            normed=False,
            filename='buckets_time_equiwidth_histogram.png')
        plot_distribution(
            Bucket.to_histogram_values(equidepth_time_buckets),
            "Number of images per bucket (bucketed by Time: equidepth)",
            nbins=5,
            set_lims=False,
            normed=False,
            filename='buckets_time_equidepth_histogram.png')

    return (difficulty_buckets,
            equiwidth_time_buckets,
            equidepth_time_buckets)

if __name__ == '__main__':
    assign_buckets(serialize=True, plot=True)
