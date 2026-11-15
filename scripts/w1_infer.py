"""Frozen W1 V1 dense TTA inference from official JPEGs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'src')]


def install_dense(rf, enabled):
    import inspect
    import solar_final_infer as parent
    source = inspect.getsource(parent.dense)
    needle = 'pred = model(torch.from_numpy(np.ascontiguousarray(image[None, None])).to(device)).float().sigmoid()[0].cpu().numpy()'
    replacement = '''tensor = torch.from_numpy(np.ascontiguousarray(image[None, None])).to(device)
            if row['arm'] == 'exp_long2x' and W1_TTA:
                total = np.zeros((2, 2048, 2048), np.float32)
                for dims in ((), (3,), (2,), (2, 3)):
                    value = model(tensor.flip(dims) if dims else tensor).float().sigmoid()
                    value = (value.flip(dims) if dims else value)[0].cpu().numpy()
                    assert np.isfinite(value).all() and value.shape == total.shape
                    total += value
                pred = total / 4
            else:
                pred = model(tensor).float().sigmoid()[0].cpu().numpy()'''
    assert source.count(needle) == 1
    source = source.replace(needle, replacement).replace('device=device, threads=THREADS)', 'device=device, threads=THREADS, w1_tta=W1_TTA)')
    parent.W1_TTA = enabled
    exec(compile(source, 'w1_dense_float32_mean4', 'exec'), parent.__dict__)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--final', choices=['rf_r34_tta'], default='rf_r34_tta')
    parser.add_argument('--dense-long2x-tta', action='store_true',
                        help='Four-flip long2x dense TTA; also implied by rf_r34_tta.')
    parser.add_argument('--stage', choices=['dense', 'support', 'score', 'refiner', 'all'], default='all')
    args = parser.parse_args()
    if args.stage == 'all':
        import subprocess
        for stage in ('dense', 'support', 'score', 'refiner'):
            command = [sys.executable, str(Path(__file__).resolve())]
            for key in ('images', 'models', 'cache', 'output', 'device', 'final'):
                command += ['--' + key, str(getattr(args, key))]
            subprocess.run(command + ['--stage', stage, '--dense-long2x-tta'], check=True)
        return
    import rf_infer as rf
    args.cache.mkdir(parents=True, exist_ok=True)
    recipe = dict(final_family='W1_V1', views=[[], [3], [2], [2, 3]],
                  averaging='numpy float32 sequential mean4 then rint u8; mean5 quantized folds')
    marker = args.cache / 'W1_RECIPE.json'
    if marker.exists():
        assert json.loads(marker.read_text()) == recipe
    rf.write(marker, recipe)
    if args.stage == 'dense':
        install_dense(rf, True)
    output = args.cache / ('rf_refiner_r34.csv' if args.stage == 'refiner' else 'rf.csv')
    sys.argv = [str(ROOT / 'scripts/rf_infer.py'), '--images', str(args.images),
                '--models', str(args.models), '--cache', str(args.cache),
                '--output', str(output), '--device', args.device, '--stage', args.stage,
                '--final', 'rf_refiner_r34' if args.stage == 'refiner' else 'rf']
    rf.main()
    if args.stage == 'refiner':
        shutil.copy2(output, args.cache / 'rf_r34_tta.csv')
        if output.resolve() != args.output.resolve():
            shutil.copy2(output, args.output)


if __name__ == '__main__':
    main()
