import matplotlib.pyplot as plt
from PIL import Image

files = [
    ("neumf_baseline.png", "(a) NeuMF / NCF baseline training"),
    ("lightgcn_baseline.png", "(b) LightGCN baseline training"),
    ("sasrec_baseline.png", "(c) SASRec baseline training"),
]

fig, axes = plt.subplots(3, 1, figsize=(14, 15))

for ax, (path, title) in zip(axes, files):
    image = Image.open(path)
    ax.imshow(image)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "Baseline_Model_Training_Evidence.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved: Baseline_Model_Training_Evidence.png")
