import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from recbole.config import Config
from recbole.data import create_dataset

config = Config(
    model="NeuMF",
    dataset="lastfm-1k",
    config_dict={
        "data_path": "dataset/",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
    },
)

print("Loading dataset through RecBole ...")
ds = create_dataset(config)
print(ds)
