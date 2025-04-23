import numpy as np
import matplotlib.pyplot as plt

def preprocess_demos(demos):
    # find the min length
    min_length = min(demo.shape[0] for demo in demos)

    # truncate demos to the min length
    processed_demos = [demo[:min_length] for demo in demos]
    return processed_demos

def plot_demos_with_mean_and_variance(demos):

    # demos = [demo * 10 for demo in demos]  # changes the scale of the y axis, which makes the variance appear larger...
    all_data = np.stack(demos, axis=0)

    # compute mean and variance
    mean_xyz = np.mean(all_data, axis=0) 
    std_xyz = np.std(all_data, axis=0)
    variance_xyz = np.var(all_data, axis=0)

    labels = ['x', 'y', 'z']
    colors = ["red", "blue", "green"]

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.2)

    # plot each demo
    for i, demo in enumerate(demos):
        for j, label in enumerate(labels):
            ax.plot(demo[:, j], label=f'Demo {i+1} {label}', color=colors[j], alpha=0.5)
            '''
            ax.fill_between(
            range(demo.shape[0]),
            - per_trajectory_variance[i],
            + per_trajectory_variance[i],
            color=colors[j],
            alpha=0.2,
            label=f'{label} variance'
            )   
            '''

    # plot mean and variance
    for j, label in enumerate(labels):
        ax.plot(mean_xyz[:, j], label=f'{label} mean', color=colors[j], linestyle='--', linewidth=2)
        ax.fill_between(
            range(mean_xyz.shape[0]),
            mean_xyz[:, j] - 2 * std_xyz[:, j],
            mean_xyz[:, j] + 2 * std_xyz[:, j],
            color=colors[j],
            alpha=0.2,
            label=f'{label} standard deviation'  # std or variance
        )

    ax.set_ylabel('Position') # x 10^1
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True,shadow=True,fontsize=8,ncol=3)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_facecolor("#F9F9F9")
    plt.title("Demonstrations with Mean and Standard Deviation")
    plt.show()

if __name__ == '__main__':
    demo_files = [
        '/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/fetch_table_demos/xyz data/full_tasks/fetch_recorded_demo_1730997119.txt',
        '/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/fetch_table_demos/xyz data/full_tasks/fetch_recorded_demo_1730997530.txt',
        '/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/fetch_table_demos/xyz data/full_tasks/fetch_recorded_demo_1730997735.txt',
        '/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/fetch_table_demos/xyz data/full_tasks/fetch_recorded_demo_1730997956.txt'
    ]

    demos_data = [np.loadtxt(file) for file in demo_files]
    processed_demos = preprocess_demos(demos_data)
    plot_demos_with_mean_and_variance(processed_demos) 