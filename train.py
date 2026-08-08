import argparse
import sys
import types
import numpy as np
import torch

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

if "ray" not in sys.modules:
    fake_ray = types.ModuleType("ray")
    fake_ray_tune = types.ModuleType("ray.tune")
    fake_ray.tune = fake_ray_tune
    sys.modules["ray"] = fake_ray
    sys.modules["ray.tune"] = fake_ray_tune

# PyTorch 2.6+ changed torch.load's default to weights_only=True, which
# RecBole 1.2.1's checkpoint format is not compatible with. Safe to
# disable here since checkpoints are ones we trained ourselves locally.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from recbole.quick_start import run_recbole

CONFIG_MAP = {
    "NCF": "ncf.yaml",
    "NeuMF": "ncf.yaml",
    "LightGCN": "lightgcn.yaml",
    "SASRec": "sasrec.yaml",
}

MODEL_NAME_MAP = {
    "NCF": "NeuMF",
    "NeuMF": "NeuMF",
    "LightGCN": "LightGCN",
    "SASRec": "SASRec",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(CONFIG_MAP.keys()))
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    config_path = CONFIG_MAP[args.model]
    model_name = MODEL_NAME_MAP[args.model]

    print(f"Training {model_name} on {args.dataset} using {config_path} ...")
    result = run_recbole(
        model=model_name,
        dataset=args.dataset,
        config_file_list=[config_path],
        config_dict={"data_path": "dataset/"},
    )
    print("\n=== Training complete ===")
    print("Best valid result:", result["best_valid_result"])
    print("Test result:", result["test_result"])


if __name__ == "__main__":
    main()
