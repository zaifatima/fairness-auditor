import numpy as np
import torch
import scipy.sparse

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

if not hasattr(scipy.sparse.dok_matrix, "_update"):
    scipy.sparse.dok_matrix._update = scipy.sparse.dok_matrix.update

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_model, get_trainer

config = Config(
    model="LightGCN",
    dataset="ml-32m-50k",
    config_file_list=["lightgcn.yaml"],
    config_dict={"data_path": "dataset/"},
)
init_seed(config["seed"], config["reproducibility"])
init_logger(config)

print("Loading dataset ...", flush=True)
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

print("Building model and loading checkpoint ...", flush=True)
model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
checkpoint = torch.load("saved/LightGCN-ml-32m-50k-latest.pth", map_location=config["device"], weights_only=False)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

print("Starting evaluation -- this will take a while, do not interrupt.", flush=True)
result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)

print("\n=== Evaluation complete ===", flush=True)
print("Test result:", result, flush=True)
