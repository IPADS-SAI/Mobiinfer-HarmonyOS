"""Compare the repacked complete MNN vision + CPU LLM with its source package.

Both executions use host CPU kernels. This does not validate HiAI compilation,
OMC execution, or accuracy equivalence between W8 MNN and W4 offline vision.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def verify(args):
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for copies in (1, 2):
        directories = []
        for label, config in [('source', args.source / 'config.json'),
                              ('repacked', args.package / 'config.json')]:
            directory = args.output / f'{label}_{copies}'
            directories.append(directory)
            log_path = args.output / f'{label}_{copies}.log'
            with log_path.open('w') as log:
                result = subprocess.run([str(args.driver), str(config), str(args.image),
                                         str(directory), str(copies)], stdout=log, stderr=log)
            if result.returncode:
                raise RuntimeError(f'{label}, copies={copies}: exit {result.returncode}; see {log_path}')
        source, repacked = directories
        tokens = np.fromfile(source / 'tokens.bin', np.int32)
        np.testing.assert_array_equal(tokens, np.fromfile(repacked / 'tokens.bin', np.int32))
        before = np.fromfile(source / 'logits.bin', np.float32)
        after = np.fromfile(repacked / 'logits.bin', np.float32)
        assert before.size and np.isfinite(before).all() and np.isfinite(after).all()
        np.testing.assert_array_equal(before, after)
        generated = np.fromfile(source / 'generated.bin', np.int32)
        np.testing.assert_array_equal(generated, np.fromfile(repacked / 'generated.bin', np.int32))
        results.append({'images': copies, 'tokens': len(tokens), 'image_tokens': int((tokens == 151655).sum()),
                        'prefill_logits_max_abs': float(np.max(np.abs(after - before))),
                        'generated_tokens_exact': True, 'generated_tokens': generated.tolist(),
                        'answer_prefix': (repacked / 'answer.txt').read_text()})
    report = {'mnn_visual_source': str(args.source), 'repacked': str(args.package),
              'reference': 'same supplied MNN weights, complete visual pipeline and CPU LLM',
              'cases': results, 'hiai_executed': False, 'omc_executed': False,
              'full_npu_accuracy_equivalence_verified': False}
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('driver', 'source', 'package', 'image', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    verify(p.parse_args())
