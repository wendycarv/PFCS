import numpy as np
import matplotlib.pyplot as plt
import h5py

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18

# grab xyz data from file
def process_demo_xyz(xyz_file, h5_file):
    xyz_data = np.loadtxt(xyz_file)
    return xyz_data

# grab specific group from h5 file (smoothed here)
def return_data(file_path):
    with h5py.File(file_path, 'r') as hf:
        if "smoothed" not in hf:
            print("No 'smoothed' group found in the file.")
            return

        # need to find min/max values of y for flipping
        all_y_values = []
        for key in hf["smoothed"].keys():
            y_data = np.array(hf[f"smoothed/{key}/y"])
            all_y_values.extend(y_data)
        
        y_min, y_max = min(all_y_values), max(all_y_values)

        # iterate over each stroke
        for key in sorted(hf["smoothed"].keys(), key=int):
            x_data = np.array(hf[f"smoothed/{key}/x"])
            y_data = np.array(hf[f"smoothed/{key}/y"])
            t_data = np.array(hf[f"smoothed/{key}/t"])

            # sort by time
            sorted_indices = np.argsort(t_data)
            x_data = x_data[sorted_indices]
            y_data = y_data[sorted_indices]

            # reverse according to max y value
            y_data = y_max - (y_data - y_min)

            return x_data, y_data

def cos_similarity(a, b):
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0
    return num / denom

class Correlation_Segmentation(object):
    def __init__(self, demo, sub_tasks, metric='SSE'):
        self.full_demo = demo
        (self.n_pts, self.n_dims) = np.shape(self.full_demo)
        self.sub_tasks = sub_tasks
        self.M = len(self.sub_tasks)
        self.sim_metric = metric
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y']

    def segment(self):
        self.Q = -float('inf') * np.ones((self.n_pts, self.M))
        for i in range(self.M):
            t_i = len(self.sub_tasks[i])
            for j in range(self.n_pts - t_i + 1):
                if self.sim_metric == 'CCS':
                    sim_ij = sum([np.dot(self.sub_tasks[i][m, :], self.full_demo[m+j, :]) for m in range(t_i)])
                elif self.sim_metric == 'SSE':
                    sim_ij = sum([-(np.linalg.norm(self.sub_tasks[i][m, :] - self.full_demo[m+j, :])**2) for m in range(t_i)])
                elif self.sim_metric == 'COS':
                    sim_ij = sum([cos_similarity(self.sub_tasks[i][m+1, :] - self.sub_tasks[i][m, :], self.full_demo[m+j+1, :] - self.full_demo[m+j, :]) for m in range(t_i-1)])
                else:
                    print('Similarity not implemented!')
                    sim_ij = 0
                self.Q[j:j+t_i, i] = np.maximum(self.Q[j:j+t_i, i], sim_ij)
        self.Z = np.argmax(self.Q, axis=1)
        indices = []
        for i in range(len(self.Z)):
            if (i == 0):
                prev_class = self.Z[i]
                indices.append(i)
            curr_class = self.Z[i]
            if (curr_class != prev_class):
                prev_class = curr_class
                indices.append(i)
        return self.Z, indices

    def plot(self):
        plt.figure()
        if self.n_dims == 1:
            plt.plot(self.full_demo, 'k')
            for i in range(self.n_pts):
                plt.plot(i, self.full_demo[i, 0], self.colors[self.Z[i]] + '.', ms=10)
        elif self.n_dims == 2:
            plt.plot(self.full_demo[:, 0], self.full_demo[:, 1], 'k')
            for i in range(self.n_pts):
                plt.plot(self.full_demo[i, 0], self.full_demo[i, 1], self.colors[self.Z[i]] + '.', ms=10)
        plt.show()

def plot_segmented_data(classes_xyz, full_task_xyz, Z, predictions, files, metric):
    colors = ['r', 'g', 'c']
    full_x_data, full_y_data = [], []
    
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_title('Original')
    
    # subject to change
    h5_path = "/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/cdog/h5 files/cdog.h5"
    with h5py.File(h5_path, 'r') as hf:
        all_y_values = []
        for key in hf["smoothed"].keys():
            y_data = np.array(hf[f"smoothed/{key}/y"])
            all_y_values.extend(y_data)

        y_min, y_max = min(all_y_values), max(all_y_values)

        for key in sorted(hf["smoothed"].keys(), key=int): 
            x_data = np.array(hf[f"smoothed/{key}/x"])
            y_data = np.array(hf[f"smoothed/{key}/y"])
            t_data = np.array(hf[f"smoothed/{key}/t"])

            sorted_indices = np.argsort(t_data)
            x_data = x_data[sorted_indices]
            y_data = y_data[sorted_indices]
            y_data = y_max - (y_data - y_min)

            full_x_data.extend(x_data)
            full_y_data.extend(y_data)
            full_x_data.append(np.nan)
            full_y_data.append(np.nan)

            ax1.plot(x_data, y_data, linewidth=5, color="black", label="Original Data")

    gt_indices = [0, 988, 1788, 2999]  # ground truth indices here, subject to change
    gt_x = [full_x_data[i] for i in gt_indices if i < len(full_x_data)]
    gt_y = [full_y_data[i] for i in gt_indices if i < len(full_y_data)]
    ax1.plot(gt_x, gt_y, '*', markersize=12, color='red', label='Ground Truth')
    ax1.tick_params(axis='both', which='both', left=False, bottom=False, top=False, length=0, labelleft=False, labelbottom=False)
    ax1.legend()
    plt.savefig("dog-orig.png", dpi=300)

    plt.figure(figsize=(9, 5))
    for i, file in enumerate(files):
        ax2 = plt.subplot(1, len(files), i+1)
        ax2.set_title(f'Class {i+1}')
        with h5py.File(f"/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/dog/h5 files/{file}.h5", 'r') as hf:
            all_y_values = []
            for key in hf["smoothed"].keys():
                y_data = np.array(hf[f"smoothed/{key}/y"])
                all_y_values.extend(y_data)
            y_min, y_max = min(all_y_values), max(all_y_values)
            for key in sorted(hf["smoothed"].keys(), key=int):
                x_data = np.array(hf[f"smoothed/{key}/x"])
                y_data = np.array(hf[f"smoothed/{key}/y"])
                t_data = np.array(hf[f"smoothed/{key}/t"])
                sorted_indices = np.argsort(t_data)
                x_data = x_data[sorted_indices]
                y_data = y_data[sorted_indices]
                y_data = y_max - (y_data - y_min)
                ax2.plot(x_data, y_data, linewidth=5, color=colors[i])
                ax2.tick_params(axis='both', which='both', left=False, bottom=False, top=False, length=0, labelleft=False, labelbottom=False)
    plt.savefig("dog-classes.png", dpi=300)

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.set_title(f'{metric} Segmentation')
    letters = ['d','o', 'g']
    used_labels = set()
    for i in range(len(predictions)):
        start_idx = predictions[i]
        end_idx = predictions[i + 1] if i != len(predictions)-1 else len(full_task_xyz)
        segment_class = Z[start_idx]
        class_letter = letters[segment_class]
        label = f"Class '{class_letter}'"
        if label not in used_labels:
            ax3.plot(full_x_data[start_idx:end_idx], full_y_data[start_idx:end_idx], linewidth=5, color=colors[segment_class], label=label)
            used_labels.add(label)
        else:
            ax3.plot(full_x_data[start_idx:end_idx], full_y_data[start_idx:end_idx], linewidth=5, color=colors[segment_class])
    ax3.tick_params(axis='both', which='both', left=False, bottom=False, top=False, length=0, labelleft=False, labelbottom=False)
    ax3.legend()
    plt.savefig(f"dog-{metric}.png", dpi=300)
    plt.show()

'''
def main():
    # pre-defined subtasks 
    files = ["class1", "class2", "class3"]

    # base directories with xyz data in .txt files and .h5 files
    xyz_dir = "/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/cdog/xy data/"
    h5_dir = "/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/cdog/h5 files/"
    metric = 'CCS'

    classes_xyz = [process_demo_xyz(xyz_dir + "noncursive/" + file + ".txt", h5_dir + "noncursive/" + file + ".h5") for file in files]
    
    # place full task file name here
    full_task_xyz = process_demo_xyz(xyz_dir + "cdog.txt", h5_dir + "cdog.h5")

    CS = Correlation_Segmentation(full_task_xyz, classes_xyz, metric=metric)
    Z, predictions = CS.segment()

    plot_segmented_data(classes_xyz, full_task_xyz, Z, predictions, files, metric)

if __name__ == '__main__':
    main()
'''