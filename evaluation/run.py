#!/usr/bin/env python3
"""Table 2/3 evaluation using pinned VLMEvalKit model/dataset adapters."""
import argparse
import contextlib
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

from common import (HERE, PROTOCOL, Capture, atomic_json, build_message,
                    checked_manifest, digest, normalize_metrics)


def arguments(default_dataset=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--toolkit", type=Path, default=HERE / "vendor" / "VLMEvalKit")
    p.add_argument("--dataset", choices=["all", *PROTOCOL["datasets"]], default=default_dataset or "all")
    p.add_argument("--model", choices=["all", *PROTOCOL["models"]], default="all")
    p.add_argument("--mode", choices=["prepare", "infer", "eval", "all"], default="all")
    p.add_argument("--output", type=Path, default=HERE / "outputs")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--model-path", help="Local snapshot or HF model ID; requires one --model")
    p.add_argument("--limit", type=int, default=0, help="Smoke test only; excluded from paper summary")
    p.add_argument("--seed", type=int, default=PROTOCOL["seed"])
    p.add_argument("--dry-run", action="store_true", help="Print planned jobs without loading dependencies/data/models")
    return p.parse_args()


def validate_toolkit(path):
    path = path.resolve()
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if commit != PROTOCOL["vlmevalkit_commit"]:
        raise ValueError(f"Expected toolkit {PROTOCOL['vlmevalkit_commit']}, found {commit}")
    changes = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    if changes.strip():
        raise ValueError("Toolkit has tracked changes; use the clean pinned checkout.")
    sys.path.insert(0, str(path))
    return commit


def versions():
    result = {"python": sys.version}
    for package in ("torch", "transformers", "qwen-vl-utils", "pandas", "numpy", "accelerate", "flash-attn"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def run_pair(args):
    # Official TreeBench evaluator requires XLSX. Prevent ambient format overrides.
    os.environ["PRED_FORMAT"] = "xlsx"
    os.environ["EVAL_FORMAT"] = "csv"
    if args.data_root:
        os.environ["LMUData"] = str(args.data_root.resolve())
    commit = validate_toolkit(args.toolkit)
    from vlmeval.dataset import build_dataset
    from vlmeval.config import supported_VLM
    from vlmeval.smp import dump
    import pandas as pd

    dataset_name = PROTOCOL["datasets"][args.dataset]
    model_name = PROTOCOL["models"][args.model]
    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset not registered: {dataset_name}")
    frame = dataset.data.copy()
    required = {"index", "question", "answer", "category"}
    if dataset_name == "TreeBench":
        required.add("target_instances")
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing fields: {required - set(frame.columns)}")
    if frame['index'].isna().any() or frame['index'].astype(str).duplicated().any():
        raise ValueError("Dataset index must be nonempty and unique")
    if frame['answer'].isna().any():
        raise ValueError("Dataset contains missing answers")
    total = len(frame)
    if dataset_name == "TreeBench" and total != 405:
        raise ValueError(f"Expected full TreeBench of 405 rows, found {total}")
    if args.limit:
        frame = frame.head(args.limit).copy()
    dataset_path = Path(dataset.data_path)
    dataset_info = {
        "name": dataset_name, "rows": total, "selected_rows": len(frame),
        "source_sha256": digest(dataset_path),
        "categories": {str(k): int(v) for k, v in dataset.data['category'].value_counts().items()},
    }
    output = args.output.resolve() / ("smoke" if args.limit else "full") / dataset_name / args.model
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        # Materialize images and check file headers without loading a model.
        from PIL import Image
        for _, row in frame.iterrows():
            images = dataset.dump_image(row)
            images = [images] if isinstance(images, str) else images
            if not images:
                raise ValueError(f"No image for {row['index']}")
            for path in images:
                with Image.open(path) as image:
                    image.verify()
        atomic_json(output / "data_check.json", dataset_info)
        print(json.dumps(dataset_info, ensure_ascii=False, indent=2))
        return
    factory = supported_VLM[model_name]
    model_kwargs = dict(factory.keywords)
    if args.model_path:
        model_kwargs["model_path"] = args.model_path
    # Resolve HF snapshot revision once per job; load locally to avoid revision drift.
    # In eval-only mode, existing identity supplies the snapshot and dependencies.
    manifest_path = output / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if args.mode == "eval":
        if old is None:
            raise ValueError("No completed inference manifest. Run infer/all first.")
        if old['dataset'] != dataset_info or old['toolkit_commit'] != commit or old['seed'] != args.seed:
            raise ValueError("Dataset/toolkit/seed differs from saved inference")
        if args.model_path and args.model_path != old['requested_model_path']:
            raise ValueError("Model path differs from saved inference")
        identity = old
    else:
        requested_path = model_kwargs['model_path']
        local = Path(requested_path).expanduser()
        if local.is_dir():
            resolved = local.resolve()
            # Hash model files for local-directory identity, including weights.
            files = sorted(p for p in resolved.rglob('*') if p.is_file() and '.cache' not in p.parts)
            model_identity = {str(p.relative_to(resolved)): digest(p) for p in files}
        else:
            from huggingface_hub import snapshot_download
            revision = old.get('model_revision') if old else None
            resolved = Path(snapshot_download(requested_path, revision=revision))
            model_identity = resolved.name
        identity = {
            "dataset": dataset_info, "model": model_name, "model_key": args.model,
            "requested_model_path": requested_path, "model_revision": model_identity if isinstance(model_identity, str) else None,
            "local_model_files": model_identity if isinstance(model_identity, dict) else None,
            "registry_kwargs": model_kwargs, "toolkit_commit": commit,
            "seed": args.seed, "limit": args.limit, "judge": PROTOCOL['judge'],
            "versions": versions(), "runner_sha256": digest(Path(__file__)),
            "common_sha256": digest(HERE / 'common.py'), "protocol_sha256": digest(HERE / 'protocol.json'),
        }
        checked_manifest(manifest_path, identity)
        model_kwargs['model_path'] = str(resolved)

    records_dir = output / 'records'
    records_dir.mkdir(exist_ok=True)
    model = None
    records = []
    for position, (_, row) in enumerate(frame.iterrows()):
        record_path = records_dir / f'{position:06d}.json'
        idx = str(row['index'])
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding='utf-8'))
            if record['index'] != idx or not isinstance(record['prediction'], str):
                raise ValueError(f'Invalid checkpoint: {record_path}')
        else:
            if args.mode == 'eval':
                raise ValueError(f'Missing prediction for {idx}; resume inference first')
            if model is None:
                import random
                import numpy as np
                import torch
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                model = factory.func(**model_kwargs)
                model.set_dump_image(dataset.dump_image)
                atomic_json(output / 'effective_model.json', {
                    'class': type(model).__name__, 'generate_kwargs': model.generate_kwargs,
                    'system_prompt': getattr(model, 'system_prompt', None),
                    'min_pixels': getattr(model, 'min_pixels', None),
                    'max_pixels': getattr(model, 'max_pixels', None),
                    'custom_prompt': model.use_custom_prompt(dataset_name),
                })
            # Deterministic per-row RNG makes resumed runs independent of skipped rows.
            import random
            import numpy as np
            import torch
            row_seed = args.seed + position
            random.seed(row_seed); np.random.seed(row_seed); torch.manual_seed(row_seed)
            torch.cuda.manual_seed_all(row_seed)
            message = build_message(model, dataset, row)
            try:
                with Capture(model) as capture:
                    prediction = model.generate(message=message, dataset=dataset_name)
                if not isinstance(prediction, str) or not prediction.strip():
                    raise ValueError('Model returned an empty/non-text prediction')
                if 'Failed to obtain answer' in prediction:
                    raise ValueError('Model adapter returned a failure marker')
                record = {
                    'index': idx, 'question': str(row['question']), 'answer': str(row['answer']),
                    'category': str(row['category']), 'prediction': prediction,
                    'raw_decoded': capture.decoded, 'chat_templates': capture.templates,
                    'prompt_messages': message, 'row_seed': row_seed,
                }
                atomic_json(record_path, record)
            except Exception:
                atomic_json(output / 'last_failure.json', {'index': idx, 'traceback': traceback.format_exc()})
                raise
        records.append(record)
        print(f'{model_name}/{dataset_name}: {position + 1}/{len(frame)}', flush=True)

    with open(output / 'predictions.jsonl', 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    if args.mode == 'infer':
        return
    # Fresh scoring directory avoids stale upstream judge/result caches.
    score_dir = output / 'scores' / uuid.uuid4().hex
    score_dir.mkdir(parents=True)
    predictions = frame.drop(columns=['image'], errors='ignore').copy()
    predictions['prediction'] = [r['prediction'] for r in records]
    prediction_path = score_dir / f'{model_name}_{dataset_name}.xlsx'
    dump(predictions, str(prediction_path))
    rating = dataset.evaluate(str(prediction_path), model=PROTOCOL['judge'], nproc=1)
    raw_rating = rating.to_dict('records') if isinstance(rating, pd.DataFrame) else rating
    atomic_json(score_dir / 'official_rating.json', raw_rating)
    metrics = normalize_metrics(dataset_name, rating)
    atomic_json(output / 'summary.json', {
        'dataset': dataset_name, 'model': args.model, 'metrics': metrics,
        'rows': len(frame), 'expected_rows': total, 'complete': not args.limit and len(frame) == total,
        'scoring_path': str(score_dir), 'judge': PROTOCOL['judge'],
        'scoring_versions': versions(), 'toolkit_commit': commit,
        'manifest_sha256': digest(manifest_path),
    })
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main(default_dataset=None):
    args = arguments(default_dataset)
    if args.limit < 0 or args.seed < 0 or args.seed >= 2**32 - 1000000:
        raise SystemExit('Invalid --limit or --seed')
    if args.model_path and args.model == 'all':
        raise SystemExit('--model-path requires --model base or treevgr')
    datasets = list(PROTOCOL['datasets']) if args.dataset == 'all' else [args.dataset]
    models = list(PROTOCOL['models']) if args.model == 'all' else [args.model]
    if args.mode == 'prepare':
        models = ['base']  # dataset checks are shared
    if args.dry_run:
        print(json.dumps({'toolkit_commit': PROTOCOL['vlmevalkit_commit'],
                          'jobs': [{'dataset': PROTOCOL['datasets'][d], 'model': PROTOCOL['models'][m], 'mode': args.mode}
                                   for d in datasets for m in models]}, indent=2))
        return
    if len(datasets) * len(models) > 1:
        for d in datasets:
            for m in models:
                command = [sys.executable, str(HERE / 'run.py'), '--dataset', d, '--model', m,
                           '--toolkit', str(args.toolkit.resolve()), '--output', str(args.output.resolve()),
                           '--mode', args.mode, '--limit', str(args.limit), '--seed', str(args.seed)]
                if args.data_root:
                    command += ['--data-root', str(args.data_root.resolve())]
                if args.model_path:
                    command += ['--model-path', args.model_path]
                subprocess.run(command, check=True)
        return
    args.model = models[0]
    log_dir = args.output.resolve() / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    log_path = log_dir / f'{args.dataset}-{args.model}-{stamp}.log'
    print(f'Running {args.dataset}/{args.model}; detailed output: {log_path}', flush=True)
    class Tee:
        def __init__(self, console, log): self.console, self.log = console, log
        def write(self, text): self.console.write(text); self.log.write(text); return len(text)
        def flush(self): self.console.flush(); self.log.flush()
    with log_path.open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                run_pair(args)
            except Exception:
                traceback.print_exc()
                raise SystemExit(1)


if __name__ == '__main__':
    main()
