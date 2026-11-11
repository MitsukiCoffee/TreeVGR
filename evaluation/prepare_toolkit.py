"""Download the pinned upstream evaluator; does not install packages or models."""
import argparse
from pathlib import Path
import subprocess
from common import HERE, PROTOCOL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, default=HERE / 'vendor' / 'VLMEvalKit')
    args = p.parse_args()
    path = args.destination.resolve()
    commit = PROTOCOL['vlmevalkit_commit']
    if path.exists():
        print(f'Using existing VLMEvalKit: {path}')
        return
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', '--depth', '1',
                        'https://github.com/open-compass/VLMEvalKit.git', str(path)], check=True)
        subprocess.run(['git', '-C', str(path), 'fetch', '--depth', '1', 'origin', commit], check=True)
        subprocess.run(['git', '-C', str(path), 'checkout', '--detach', commit], check=True)
    print(f'VLMEvalKit: {path}\nCommit: {commit}')


if __name__ == '__main__':
    main()
