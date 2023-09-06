from util.plot_utils import plot_logs
from pathlib import Path

log_directory = [Path('runs2/')]

fields_of_interest = ('mAP',)


if __name__ == "__main__":
    
    plot_logs(log_directory, fields_of_interest)