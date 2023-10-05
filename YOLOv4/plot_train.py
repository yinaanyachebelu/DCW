import glob
import math
import os
import random
from copy import copy
from pathlib import Path

from utils.general import *
from utils.plots import *

def get_args_parser():
    parser = argparse.ArgumentParser(
        'yolo plotting', add_help=False)
    parser.add_argument('--save_dir', default='', type=str)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(
        'yolo plotting', parents=[get_args_parser()])
        args = parser.parse_args()
        
        plot_results_overlay2(save_dir=args.save_dir, start=0, stop=0)
