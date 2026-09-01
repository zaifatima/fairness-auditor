import argparse
import os
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

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_model, get_trainer

CONFIG_MAP = {
    "NCF": "ncf.yaml", "NeuMF": "ncf.yaml",
    "LightGCN": "lightgcn.yaml", "SASRec": "sasrec.yaml",
}
MODEL_NAME_MAP = {
    "NCF": "NeuMF", "NeuMF": "NeuMF",
    "LightGCN": "LightGCN", "SASRec": "SASRec",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(CONFIG_MAP.keys()))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--resume", default=None, help="Path to a -latest.pth checkpoint")
    args = parser.parse_args()

    config_path = CONFIG_MAP[args.model]
    model_name = MODEL_NAME_MAP[args.model]

    config = Config(
        model=model_name, dataset=args.dataset,
        config_file_list=[config_path],
        config_dict={"data_path": "dataset/"},
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    os.makedirs("saved", exist_ok=True)
    latest_path = f"saved/{model_name}-{args.dataset}-latest.pth"
    best_path = f"saved/{model_name}-{args.dataset}-best.pth"

    start_epoch = 0
    best_valid_score = float("-inf")
    best_valid_result = None

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=config["device"])
        model.load_state_dict(ckpt["state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_valid_score = ckpt.get("best_valid_score", float("-inf"))
        best_valid_result = ckpt.get("best_valid_result", None)
        print(f"Resumed at epoch {start_epoch}, best_valid_score so far: {best_valid_score}")

    eval_step = config["eval_step"] or 1
    total_epochs = config["epochs"]

    for epoch_idx in range(start_epoch, total_epochs):
        train_loss = trainer._train_epoch(train_data, epoch_idx, show_progress=config["show_progress"])
        print(f"epoch {epoch_idx} training loss: {train_loss}", flush=True)

        # Unconditional save — every single epoch, regardless of validation result.
        torch.save({
            "epoch": epoch_idx,
            "state_dict": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "best_valid_score": best_valid_score,
            "best_valid_result": best_valid_result,
        }, latest_path)
        print(f"Checkpoint saved (epoch {epoch_idx}): {latest_path}", flush=True)

        if (epoch_idx + 1) % eval_step == 0 or epoch_idx == total_epochs - 1:
            valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=config["show_progress"])
            print(f"epoch {epoch_idx} valid result: {valid_result}", flush=True)
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                best_valid_result = valid_result
                torch.save({"epoch": epoch_idx, "state_dict": model.state_dict()}, best_path)
                print(f"New best model saved: {best_path}", flush=True)

    print("\n=== Training complete ===")
    print("Best valid result:", best_valid_result)

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=config["device"])
        model.load_state_dict(ckpt["state_dict"])
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config["show_progress"])
    print("Test result:", test_result)


if __name__ == "__main__":
    main()
