# 수어 학습기 by Langpeu - (c) 2026 Langpeu, licensed under AGPL-3.0 (see LICENSE)
# Helper for 0_setup.bat: picks the right PyTorch build for this PC's GPU and checks the install
import subprocess, sys

BASE = 'https://download.pytorch.org/whl/'


def gpu():
    """returns (name, compute capability tuple or None) of the first NVIDIA GPU, or (None, None)"""
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=30)
        line = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ''
        if line:
            name, cc = [x.strip() for x in line.split(',', 1)]
            major, minor = (int(v) for v in cc.split('.'))
            return name, (major, minor)
    except Exception:  # noqa: BLE001
        pass
    try:  # older drivers: compute_cap query not supported
        r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0].strip(), None
    except Exception:  # noqa: BLE001
        pass
    return None, None


def torch_build():
    name, cc = gpu()
    if name is None:
        return 'cpu', name, cc
    if cc is not None and cc >= (7, 5):     # Turing (RTX 20xx) and newer, incl. RTX 50xx
        return 'cu128', name, cc
    return 'cu126', name, cc                # older NVIDIA GPUs


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'info'
    tag, name, cc = torch_build()
    if mode == 'index':
        print(BASE + tag)
    elif mode == 'info':
        if name is None:
            print('  No NVIDIA GPU found -> CPU version of PyTorch (works, but slower)')
        else:
            ccs = f'{cc[0]}.{cc[1]}' if cc else '?'
            print(f'  GPU: {name} (compute {ccs}) -> PyTorch {tag}')
    elif mode == 'check':
        import torch, cv2, mediapipe, ultralytics, PIL  # noqa: F401
        print('  torch', torch.__version__, '| opencv', cv2.__version__, '| mediapipe', mediapipe.__version__,
              '| ultralytics', ultralytics.__version__)
        if torch.cuda.is_available():
            print('  CUDA OK:', torch.cuda.get_device_name(0))
        elif name is not None:
            print('  WARNING: NVIDIA GPU found but CUDA is not usable. Update the NVIDIA driver, then run 0_setup.bat again.')
        else:
            print('  Running on CPU.')


if __name__ == '__main__':
    main()
