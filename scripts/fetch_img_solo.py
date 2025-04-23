import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.image as mpimg

plt.rcParams["font.family"] = "Times New Roman"

image_path = "/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/fetch_table_demos/long tasks/screenshots/wperson.png"
img = mpimg.imread(image_path)

# subtasks and their corresponding indices
subtasks = ["plate", "napkin", "cup", "fork", "spoon"]
#indices = [0, 1125, 2591, 3986, 5666, 7338]  
indices = [0, 25, 58, 89, 127, 163]  
colors = ['r', 'g', 'b', 'm', 'c']

fig, ax = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [4, 0.8]}) 

ax[0].imshow(img)
ax[0].axis("off")

# subtask segmentation plot
for i in range(len(subtasks)):
    ax[1].barh(0, width=indices[i+1] - indices[i], left=indices[i], color=colors[i], edgecolor="black")

ax[1].set_xlim(0, indices[-1])
ax[1].set_yticks([])
ax[1].set_title("Subtask Segmentation", fontsize=14, pad=5)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

legend_patches = [mpatches.Patch(color=colors[i], label=subtasks[i]) for i in range(len(subtasks))]
ax[1].legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.7), ncol=len(subtasks))

plt.tight_layout()
plt.savefig("fetch-solo.png", dpi=300)
plt.show()
