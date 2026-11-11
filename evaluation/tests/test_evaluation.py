import functools
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Capture, atomic_json, build_message, normalize_metrics, PROTOCOL
from summarize import collect
import run


class ProtocolTests(unittest.TestCase):
    def test_treebench_percent_scale(self):
        rating = {'Overall': .504, 'mIoU': .44, 'Perception': {'OCR': .633}, 'Reasoning': {'Ordering': .368}}
        metrics = normalize_metrics('TreeBench', rating)
        self.assertAlmostEqual(metrics['Overall'], 50.4)
        self.assertEqual(metrics['mIoU'], 44)

    def test_hr_uses_cycle_average(self):
        rating = [{'cycle': '0', 'type': 'all', 'accuracy': .99},
                  {'cycle': 'Average', 'type': 'all', 'accuracy': .771},
                  {'cycle': 'Average', 'type': 'single', 'accuracy': .903},
                  {'cycle': 'Average', 'type': 'cross', 'accuracy': .64}]
        self.assertAlmostEqual(normalize_metrics('HRBench4K', rating)['Overall'], 77.1)

    def test_vstar_categories(self):
        rating = [{'Overall': .911, 'direct_attributes': .94, 'relative_position': .87}]
        self.assertEqual(normalize_metrics('VStarBench', rating)['Spatial'], 87)
        with self.assertRaises(ValueError):
            normalize_metrics('VStarBench', rating * 2)

    def test_official_prompt_precedence(self):
        model = types.SimpleNamespace(use_custom_prompt=lambda d: True,
                                      build_prompt=lambda row, dataset: 'model')
        data = types.SimpleNamespace(dataset_name='TreeBench', build_prompt=lambda row: 'dataset')
        self.assertEqual(build_message(model, data, {}), 'model')
        data.force_use_dataset_prompt = True
        self.assertEqual(build_message(model, data, {}), 'dataset')

    def test_capture_restores_after_failure(self):
        tokenizer = types.SimpleNamespace(batch_decode=lambda *a, **kw: ['<answer>A</answer>'])
        processor = types.SimpleNamespace(tokenizer=tokenizer, apply_chat_template=lambda *a, **kw: 'prompt')
        original = tokenizer.batch_decode
        with self.assertRaises(RuntimeError):
            with Capture(types.SimpleNamespace(processor=processor)) as capture:
                self.assertEqual(tokenizer.batch_decode([]), ['<answer>A</answer>'])
                processor.apply_chat_template([])
                raise RuntimeError('test')
        self.assertIs(tokenizer.batch_decode, original)
        self.assertEqual(capture.templates, ['prompt'])

    def test_missing_runs_are_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, missing = collect(Path(tmp))
            self.assertEqual(len(missing), 8)
            self.assertIsNone(rows[0]['measured_base'])


class RunnerIntegrationTests(unittest.TestCase):
    """CPU-only orchestration test with fake model/dataset; not accuracy validation."""
    def test_run_resume_eval_and_raw_preservation(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'data.tsv'; source.write_text('fixture', encoding='utf-8')
            weights = root / 'weights'; weights.mkdir()
            (weights / 'config.json').write_text('{}', encoding='utf-8')
            calls = []
            class Dataset:
                dataset_name = 'VStarBench'
                data_path = source
                data = pd.DataFrame([{'index': i, 'question': 'q', 'answer': 'A',
                                      'category': c} for i, c in enumerate(['direct_attributes', 'relative_position'])])
                def dump_image(self, row): return ['fixture.png']
                def build_prompt(self, row): return [{'type': 'text', 'value': 'question'}]
                def evaluate(self, path, **kwargs):
                    self_test.assertEqual(kwargs['model'], 'exact_matching')
                    frame = pd.read_excel(path)
                    self_test.assertEqual(list(frame.prediction), ['A', 'A'])
                    return pd.DataFrame([{'Overall': 1., 'direct_attributes': 1., 'relative_position': 1.}])
            class Model:
                generate_kwargs = {'do_sample': False}
                def __init__(self, **kwargs):
                    calls.append('load')
                    self.processor = types.SimpleNamespace(
                        tokenizer=types.SimpleNamespace(batch_decode=lambda *a, **kw: ['reason <answer>A</answer>']),
                        apply_chat_template=lambda *a, **kw: 'rendered')
                def set_dump_image(self, f): pass
                def use_custom_prompt(self, d): return False
                def generate(self, **kwargs):
                    calls.append('generate')
                    self.processor.apply_chat_template([])
                    self.processor.tokenizer.batch_decode([])
                    return 'A'
            def module(name, **attributes):
                m = types.ModuleType(name)
                m.__dict__.update(attributes)
                return m
            fake_modules = {
                'vlmeval': module('vlmeval'),
                'vlmeval.dataset': module('vlmeval.dataset', build_dataset=lambda d: Dataset()),
                'vlmeval.config': module('vlmeval.config', supported_VLM={
                    'TreeVGR-7B': functools.partial(Model, model_path=str(weights))}),
                'vlmeval.smp': module('vlmeval.smp', dump=lambda df, path: df.to_excel(path, index=False)),
                'huggingface_hub': module('huggingface_hub'),
                'torch': module('torch', manual_seed=lambda s: None,
                                cuda=types.SimpleNamespace(manual_seed_all=lambda s: None)),
            }
            args = types.SimpleNamespace(data_root=None, toolkit=root, dataset='vstar', model='treevgr',
                                         limit=0, output=root/'outputs', mode='all', seed=0, model_path=None)
            self_test = self
            with patch.dict(sys.modules, fake_modules), patch.object(run, 'load_toolkit'):
                run.run_pair(args)
                self.assertEqual(calls, ['load', 'generate', 'generate'])
                run.run_pair(args)
                self.assertEqual(calls, ['load', 'generate', 'generate'])
                args.mode = 'eval'
                run.run_pair(args)
            directory = args.output/'full'/'VStarBench'/'treevgr'
            record = json.loads((directory/'records'/'000000.json').read_text())
            self.assertEqual(record['prediction'], 'A')
            self.assertEqual(record['raw_decoded'], ['reason <answer>A</answer>'])
            summary = json.loads((directory/'summary.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['metrics']['Overall'], 100)


if __name__ == '__main__':
    unittest.main()
