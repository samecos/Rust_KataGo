"""Rebuild the archived AOT source in a fresh directory; never replace certified assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cuda', type=Path, default=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3'))
    parser.add_argument('--ccbin', type=Path, default=Path('C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64'))
    parser.add_argument('--cutlass', type=Path, default=Path('D:/code/cutlass'))
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'source-manifest.json').read_text())
    for item in manifest['source_files']:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    nvcc = args.cuda / 'bin/nvcc.exe'
    assert sha(nvcc) == manifest['nvcc_sha256'], 'Use the recorded compiler or requalify the artifact'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('KATAGO_') and k.upper() not in ['NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS']}
    base = [str(nvcc), '-O3', '--std=c++17', '-ccbin', str(args.ccbin), '--expt-relaxed-constexpr',
            '-Xcompiler', '/Zc:preprocessor /Zc:__cplusplus /EHsc /bigobj /std:c++17', '-arch=sm_120', '-I' + str(args.cutlass / 'include')]
    probe = ROOT / 'source/qkv-immutable-params-rope-r1/probe.cu'
    exporter = ROOT / 'source/qkv-immutable-params-integration-r1/abi-export.cu'
    commands = [
        ('probe-build', base + ['-lineinfo', '-Xptxas=-v', '--keep', '--keep-dir', str(out), str(probe), '-lcublasLt', '-lcuda', '-o', str(out / 'probe.exe')]),
        ('abi-build', base + [str(exporter), '-lcublasLt', '-lcuda', '-o', str(out / 'abi-export.exe')]),
        ('abi-export', [str(out / 'abi-export.exe'), str(out / 'abi')]),
    ]
    records = []
    for name, argv in commands:
        with (out / (name + '.log')).open('xb') as log:
            done = subprocess.run(argv, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW, timeout=600)
        records.append(dict(phase=name, argv=argv, returncode=done.returncode))
        (out / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        assert done.returncode == 0, 'Failure retained in ' + str(out)
    for name in ['params-template.bin', 'abi.json']:
        assert (out / 'abi' / name).read_bytes() == (ROOT / name).read_bytes(), name
    cubin = out / 'probe.sm_120.cubin'
    # Compare complete disassembled functions, excluding file/container banners.
    sass = []
    for label, path in [('certified', ROOT / 'kernel.sm120.cubin'), ('rebuilt', cubin)]:
        done = subprocess.run([str(args.cuda / 'bin/cuobjdump.exe'), '--dump-sass', str(path)], capture_output=True, env=env, creationflags=subprocess.CREATE_NO_WINDOW, timeout=60)
        assert done.returncode == 0
        (out / (label + '.sass')).write_bytes(done.stdout)
        text = done.stdout.decode('utf-8')
        assert 'Function :' in text
        sass.append(text[text.index('Function :'):])
    assert sass[0] == sass[1], 'Rebuilt machine instructions differ; no certification transfer'
    result = dict(status='PASS_REBUILD_IDENTICAL_SASS_AND_ABI_CPU_ONLY', gpu_used=False,
                  certified_cubin_sha256=sha(ROOT / 'kernel.sm120.cubin'), rebuilt_cubin_sha256=sha(cubin),
                  identical_sass=True, identical_abi=True, installed=False,
                  note='Debug/source paths may change CUBIN bytes. Runtime accepts only its compiled artifact hash; rebuilding does not migrate or certify a plan.')
    (out / 'review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], flush=True)

if __name__ == '__main__':
    main()
