"""Compare completed runs against paper Tables 2 and 3 (percentage points)."""
import argparse
import csv
import json
from pathlib import Path
from common import HERE, PROTOCOL


def collect(output):
    rows, missing = [], []
    for dataset, expected in PROTOCOL['paper'].items():
        actual = {}
        for model in PROTOCOL['models']:
            directory = output / 'full' / dataset / model
            path = directory / 'summary.json'
            if not path.exists():
                missing.append(f'{dataset}/{model}: not scored')
                continue
            value = json.loads(path.read_text(encoding='utf-8'))
            if not value['complete']:
                missing.append(f'{dataset}/{model}: incomplete')
                continue
            actual[model] = value['metrics']
        for metric in expected['treevgr']:
            pb = expected['base'].get(metric)
            pt = expected['treevgr'][metric]
            b = actual.get('base', {}).get(metric) if pb is not None else None
            t = actual.get('treevgr', {}).get(metric)
            rows.append({'dataset': dataset, 'metric': metric,
                         'paper_base': pb, 'measured_base': b,
                         'base_error_pp': b-pb if b is not None else None,
                         'paper_treevgr': pt, 'measured_treevgr': t,
                         'treevgr_error_pp': t-pt if t is not None else None,
                         'paper_gain_pp': pt-pb if pb is not None else None,
                         'measured_gain_pp': t-b if b is not None and t is not None else None})
    return rows, missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE / 'outputs')
    args = p.parse_args()
    rows, missing = collect(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'paper_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    headers = list(rows[0])
    lines = ['# Tables 2 and 3 comparison', '', 'All scores are percentages; differences are percentage points.', '',
             '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        values = [('--' if v is None else f'{v:.2f}' if isinstance(v, float) else str(v)) for v in row.values()]
        lines.append('| ' + ' | '.join(values) + ' |')
    if missing:
        lines += ['', 'Missing runs:', *['- ' + m for m in missing]]
    (args.output / 'paper_comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(args.output / 'paper_comparison.csv')
    if missing:
        print('Pending: ' + ', '.join(missing))


if __name__ == '__main__':
    main()
