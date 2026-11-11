"""Dependency-light protocol, checkpoint and metric helpers."""
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def normalize_metrics(dataset, rating):
    """Use upstream scores (0..1); never recompute HRBench cycle aggregation."""
    if dataset == "TreeBench":
        result = {"Overall": rating["Overall"], "mIoU": rating["mIoU"]}
        for task in ("Perception", "Reasoning"):
            result.update(rating[task])
    elif dataset == "VStarBench":
        rows = rating.to_dict("records") if hasattr(rating, "to_dict") else rating
        if len(rows) != 1:
            raise ValueError("Unexpected VStar split layout; inspect official rating before aggregating.")
        row = rows[0]
        result = {"Overall": row["Overall"], "Attr": row["direct_attributes"], "Spatial": row["relative_position"]}
    else:
        rows = rating.to_dict("records") if hasattr(rating, "to_dict") else rating
        rows = [r for r in rows if str(r["cycle"]) == "Average"]
        values = {r["type"]: r["accuracy"] for r in rows}
        result = {"Overall": values["all"], "Single": values["single"], "Cross": values["cross"]}
    out = {}
    for key, value in result.items():
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid upstream metric {key}: {value}")
        out[key] = value * 100
    return out


def build_message(model, dataset, row):
    """Same precedence as VLMEvalKit inference.py."""
    if getattr(dataset, "force_use_dataset_prompt", False):
        return dataset.build_prompt(row)
    if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset.dataset_name):
        return model.build_prompt(row, dataset=dataset.dataset_name)
    return dataset.build_prompt(row)


class Capture:
    """Observe decoded text/rendered templates without changing model output."""
    def __init__(self, model):
        self.processor = model.processor
        self.tokenizer = model.processor.tokenizer
        self.decoded = []
        self.templates = []

    def __enter__(self):
        self.decode = self.tokenizer.batch_decode
        self.template = self.processor.apply_chat_template
        def decode(*args, **kwargs):
            value = self.decode(*args, **kwargs)
            self.decoded.extend(value)
            return value
        def template(*args, **kwargs):
            value = self.template(*args, **kwargs)
            if isinstance(value, (str, list)):
                self.templates.append(value)
            return value
        self.tokenizer.batch_decode = decode
        self.processor.apply_chat_template = template
        return self

    def __exit__(self, *exc):
        self.tokenizer.batch_decode = self.decode
        self.processor.apply_chat_template = self.template
