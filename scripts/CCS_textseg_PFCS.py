import numpy as np
import matplotlib.pyplot as plt
import h5py
from PFCS import *

def main():
    # pre-defined subtasks 
    files = ["class1", "class2", "class3"]

    # base directories with xyz data in .txt files and .h5 files
    xyz_dir = "/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/cdog/xy data/"
    h5_dir = "/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/cdog/h5 files/"

    # define metric here (CCS, COS, or SSE)
    metric = 'CCS'

    classes_xyz = [process_demo_xyz(xyz_dir + "noncursive/" + file + ".txt", h5_dir + "noncursive/" + file + ".h5") for file in files]
    
    # place full task file name here
    full_task_xyz = process_demo_xyz(xyz_dir + "cdog.txt", h5_dir + "cdog.h5")

    CS = Correlation_Segmentation(full_task_xyz, classes_xyz, metric=metric)
    Z, predictions = CS.segment()

    plot_segmented_data(classes_xyz, full_task_xyz, Z, predictions, files, metric)

if __name__ == '__main__':
    main()
