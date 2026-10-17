"""Independently compare packaged MNN W8 codes/scales with supplied GPTQ files.

The original converter's JSON sidecars are used only to locate payloads. This
does not requantize, load AutoRound kernels, or run HiAI on the host.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from safetensors import safe_open


def audit(source, exported, package):
    report = {'source': str(source), 'requantized': False, 'language_matrices': 0,
              'visual_matrices': 0, 'weights_exact': 0, 'scales_exact': 0,
              'source_activation_config': json.loads((source / 'quantization_config.json').read_text()),
              'runtime_activation_equivalence_verified': False}
    seen_visual = set()
    with safe_open(source / 'model.safetensors', framework='numpy') as sf:
        keys = set(sf.keys())
        graphs = ['llm'] + [f'visual_blocks_npu_{i}' for i in range(6)]
        for graph in graphs:
            assert (exported / (graph + '.mnn')).read_bytes() == (package / (graph + '.mnn')).read_bytes(), (
                graph, 'JSON sidecar must belong to the packaged binary graph')
            data = json.loads((exported / (graph + '.mnn.json')).read_text())
            with (package / (graph + '.mnn.weight')).open('rb') as binary:
                for op in data['oplists']:
                    if op['type'] != 'Convolution':
                        continue
                    name, main = op['name'], op['main']
                    section = slice(None)
                    if graph == 'llm':
                        match = re.fullmatch(r'/layers\.(\d+)/([^/]+)/([^/]+)/Linear', name)
                        if not match:
                            continue  # floating lm_head
                        layer, module, projection = match.groups()
                        base = f'model.language_model.layers.{layer}.{module}.{projection}'
                        report['language_matrices'] += 1
                    else:
                        match = re.match(r'/blocks\.(\d+)/(self_attn|mlp)/([^/]+)/', name)
                        if not match:
                            raise ValueError('Unknown visual linear: ' + name)
                        local, module, projection = match.groups()
                        layer = int(graph.rsplit('_', 1)[1]) * 4 + int(local)
                        if module == 'self_attn' and projection in ('q_proj', 'k_proj', 'v_proj'):
                            index = ('q_proj', 'k_proj', 'v_proj').index(projection)
                            width = main['common']['outputCount']
                            section = slice(index * width, (index + 1) * width)
                            projection = 'qkv'
                        module = 'attn' if module == 'self_attn' else module
                        base = f'model.visual.blocks.{layer}.{module}.{projection}'
                        seen_visual.add((layer, match.group(3)))
                        report['visual_matrices'] += 1
                    packed = sf.get_tensor(base + '.qweight').astype(np.uint32)
                    # GPTQ packs four unsigned 8-bit codes along input channels.
                    codes = ((packed[:, None, :] >> np.array([0, 8, 16, 24], np.uint32)[None, :, None]) & 255)
                    codes = codes.reshape(-1, packed.shape[1]).T[section].astype(np.uint8)
                    assert codes.shape == (main['common']['outputCount'], main['common']['inputCount'])
                    external = main['external']
                    assert main['quanParameter']['aMaxOrBits'] == 8
                    header = external[1] - codes.nbytes
                    assert header > 0
                    binary.seek(external[0] + header)
                    assert binary.read(codes.nbytes) == codes.tobytes(), (graph, name, 'weights')
                    report['weights_exact'] += 1
                    scales = sf.get_tensor(base + '.scales').astype(np.float32).T[section]
                    assert scales.shape[1] == codes.shape[1] // 128
                    if scales.nbytes < external[2]:
                        scales = np.stack([np.zeros_like(scales), scales], axis=-1)
                    assert scales.nbytes == external[2]
                    assert binary.read(external[2]) == scales.tobytes(), (graph, name, 'scales')
                    report['scales_exact'] += 1
                    zeros = sf.get_tensor(base + '.qzeros').astype(np.uint32)
                    zero_codes = (zeros[:, :, None] >> np.array([0, 8, 16, 24], np.uint32)[None, None, :]) & 255
                    assert np.all(zero_codes == 127), 'Importer requires symmetric source zeros'
                    if base + '.g_idx' in keys:
                        groups = sf.get_tensor(base + '.g_idx')
                        assert np.array_equal(groups, np.arange(len(groups)) // 128)
    assert report['language_matrices'] == 196
    assert report['visual_matrices'] == 144 and len(seen_visual) == 144
    report['all_matrices_exact'] = report['weights_exact'] == report['scales_exact'] == 340
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'exported', 'package', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    result = audit(a.source, a.exported, a.package)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_activation_config'}))
