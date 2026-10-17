"""Compose a separate Offline NPU-2 package without re-exporting/requantizing.

Never edits either input. No phone access and no publication side effects.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def package(cpu, full, output, with_mnn_vision=False):
    cpu, full, output = map(lambda p: Path(p).resolve(), (cpu, full, output))
    if output.exists():
        raise ValueError('Output must be a new directory')
    cfg = json.loads((cpu / 'config.json').read_text())
    metadata = json.loads((cpu / cfg.get('llm_config', 'llm_config.json')).read_text())
    if (metadata.get('model_type'), metadata.get('hidden_size'), metadata.get('layer_nums'),
        metadata.get('is_mrope'), metadata.get('has_deepstack')) != ('qwen3_vl', 2048, 28, True, True):
        raise ValueError('Expected matching Mobi Qwen3-VL 2B MNN language assets')
    fc = json.loads((full / 'config.json').read_text())
    if sha(cpu / cfg.get('tokenizer_file', 'tokenizer.mtok')) != sha(full / fc['tokenizer_file']):
        raise ValueError('CPU and vision package tokenizers differ; verify source model identity')
    runtime = fc['runtime']
    if [runtime[k] for k in ('image_height', 'image_width', 'patch', 'merge', 'temporal')] != [608, 256, 16, 2, 2]:
        raise ValueError('Unsupported visual grid')
    if runtime['vision_deepstack_layers'] != [5, 11, 17]:
        raise ValueError('Unsupported DeepStack captures')
    graphs = [g for g in fc['graphs'] if g['name'] == 'vision_full']
    if len(graphs) != 1 or not graphs[0].get('external_weights'):
        raise ValueError('Requires one fused external-weight vision graph')
    graph = graphs[0]
    assets = {}

    def copy(src, relative):
        src = src.resolve()
        if not src.is_file():
            raise ValueError('Missing asset: ' + str(src))
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        digest = sha(src)
        if sha(dest) != digest:
            raise ValueError('Copy changed bytes: ' + str(src))
        assets[str(relative)] = {'bytes': src.stat().st_size, 'sha256': digest}

    # Flatten the CPU assets needed by base Llm. Optional MNN vision is retained
    # separately from the vendor's frozen W4/LUT graph and external weights.
    for key, default in [('llm_model', 'llm.mnn'), ('llm_weight', 'llm.mnn.weight'),
                         ('embedding_file', 'embeddings_bf16.bin'), ('tokenizer_file', 'tokenizer.mtok')]:
        src = cpu / cfg.get(key, default)
        copy(src, src.name)
        cfg[key] = src.name
    copy(cpu / cfg.get('llm_config', 'llm_config.json'), 'llm_config.json')
    cfg['llm_config'] = 'llm_config.json'
    for key in list(cfg):
        if key.startswith(('visual_', 'npu_', 'speculative_')) or key in ('base_dir', 'tmp_path', 'mllm_cpu_core_ids'):
            del cfg[key]
    cfg.update(backend_type='cpu', prompt_cache=False, visual_split=False,
               offline_npu2={'format': 'kirin_offline_npu2_v1', 'vision_config': 'vision/config.json'})
    cfg['mllm'] = dict(cfg.get('mllm', {}), backend_type='cpu')
    if with_mnn_vision:
        source_cfg = json.loads((cpu / 'config.json').read_text())
        chunks = source_cfg.get('visual_blocks_chunks', [])
        if not source_cfg.get('visual_split') or not chunks or not all(isinstance(v, str) for v in chunks):
            raise ValueError('Online NPU requires a complete split MNN visual model')
        models = [source_cfg.get('visual_pre_model'), *chunks, source_cfg.get('visual_post_model')]
        if not all(isinstance(v, str) and v.endswith('.mnn') for v in models):
            raise ValueError('Expected MNN visual pre, blocks and post models')
        if len({Path(v).name for v in models}) != len(models):
            raise ValueError('Visual model names collide after flattening')
        for model in models:
            source = cpu / model
            copy(source, source.name)
            copy(Path(str(source) + '.weight'), source.name + '.weight')
        cfg.update(visual_split=True, visual_pre_model=Path(models[0]).name,
                   visual_post_model=Path(models[-1]).name,
                   visual_blocks_chunks=[Path(v).name for v in chunks],
                   visual_blocks_backend_type='hiai',
                   visual_blocks_chunk_backends=['npu'] * len(chunks),
                   visual_blocks_kv_hints=False, offline_npu2_mnn_visual=True)
        # Kept for older Apps which reject an Offline NPU-2 marker in online
        # mode. All paths remain relative to the same model directory.
        online_cfg = {k: v for k, v in cfg.items()
                      if k not in ('offline_npu2', 'offline_npu2_mnn_visual')}
        (output / 'config_online.json').write_text(json.dumps(online_cfg, indent=2) + '\n')
    copy(full / graph['file'], Path('vision') / graph['file'])
    if graph.get('sha256') != assets[str(Path('vision') / graph['file'])]['sha256']:
        raise ValueError('Fused graph differs from its manifest')
    for file in graph['external_weights']['files']:
        relative = Path(graph['external_weights']['directory']) / file
        copy(full / relative, Path('vision') / relative)
    copy(full / fc['vision_constants'], 'vision/vision_constants.bin')
    vision = {'format': 'kirin_offline_npu2_vision_v1', 'vision_constants': 'vision_constants.bin',
              'runtime': runtime, 'graph': graph}
    (output / 'vision/config.json').write_text(json.dumps(vision, indent=2) + '\n')
    (output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    manifest = {'format': 'kirin_offline_npu2_manifest_v1', 'cpu_source': str(cpu),
                'vision_source': str(full), 'assets': assets,
                'quantization': {'vision': 'unchanged frozen W4A16/' +
                                 fc.get('quantization', {}).get('weight_method', 'gptq').upper() + ' group64 visual export',
                                 'language': 'unchanged source MNN quantization; not converted to W4',
                                 'kv': 'MNN CPU runtime cache; no NNRT language KV'},
                'device_validation': False}
    if with_mnn_vision:
        manifest.update(supported_modes=['offline_npu2', 'online', 'cpu'],
                        mnn_visual_source=str(cpu), requantized=False)
        manifest['quantization']['mnn_vision'] = (
            'unchanged supplied AutoRound/GPTQ W8/group128 blocks; floating pre/post and DeepStack mergers')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    readme = Path(__file__).with_name('README.md')
    if readme.is_file():
        shutil.copyfile(readme, output / 'README.md')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu', type=Path, required=True)
    p.add_argument('--full-npu', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--with-mnn-vision', action='store_true',
                   help='Include complete MNN vision assets for online NPU and CPU modes')
    a = p.parse_args()
    m = package(a.cpu, a.full_npu, a.output, a.with_mnn_vision)
    print(json.dumps({'output': str(a.output), 'asset_files': len(m['assets']),
                      'asset_bytes': sum(v['bytes'] for v in m['assets'].values())}))
