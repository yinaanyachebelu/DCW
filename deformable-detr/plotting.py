from util.plot_utils import plot_logs
from pathlib import Path

log_directory = [Path('runs_focal_laprop/')]

fields_of_interest = ('mAP', 'loss')


if __name__ == "__main__":

    plot_logs(log_directory, fields=fields_of_interest)
