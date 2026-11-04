#!/usr/bin/env python3
"""Full GDINO CPU correctness check, explicitly reduced engineering input/queries."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcd.runtime import read_runtime
from gcd.protocol import write_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference-checkpoint', type=Path)
    a = p.parse_args()
    runtime = read_runtime(a.runtime)
    sys.path.insert(0, runtime['mmdet_root'])
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmdet.utils import register_all_modules
    from mmdet.registry import MODELS, DATASETS
    from gcd.model import frozen_digest
    from gcd.peft import audit_trainables
    import gcd.model
    torch.set_num_threads(2)
    torch.manual_seed(17)
    started = time.perf_counter()
    register_all_modules()
    manifest = json.loads(Path(runtime['manifest']).read_text())
    phase = manifest['phases']['task2']
    cfg = Config.fromfile(str(ROOT / 'configs/model/gdino_t.py')).model
    for key in ['as_two_stage', 'share_pred_layer', 'num_pred_layer', 'train_cfg', 'test_cfg']:
        cfg.bbox_head.pop(key, None)
    cfg.update(type='SingleLoRAGroundingDINO', initial_checkpoint=runtime['pretrained_checkpoint'],
        lora=dict(rank=16, alpha=32, dropout=0), train_label_ids=list(range(15, 20)),
        expected_classes=phase['classes'])
    cfg.language_model.name = runtime['bert_path']
    cfg.encoder.num_cp, cfg.backbone.with_cp, cfg.num_queries = 0, False, 900
    model = MODELS.build(cfg)
    model.init_weights()
    params = audit_trainables(model)
    assert params['trainable'] == 688128
    assert len(model.lora_paths) == 24
    # Load the COMPLETE W0 first, then reduce queries only for this CPU check.
    model.num_queries = 20
    query_weight = model.query_embedding.weight[:20].detach().clone()
    model.query_embedding = torch.nn.Embedding.from_pretrained(query_weight, freeze=True)
    meta = ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'text', 'custom_entities')
    ds = DATASETS.build(dict(type='CocoDataset', ann_file=phase['smoke_eval_ann'],
        data_prefix=dict(img=phase['image_root'] + '/'), metainfo=dict(classes=tuple(phase['classes'])),
        return_classes=True, test_mode=True, pipeline=[dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True), dict(type='FixScaleResize', scale=(96, 128), keep_ratio=True),
        dict(type='PackDetInputs', meta_keys=meta)]))
    batch = model.data_preprocessor(pseudo_collate([ds[0]]), training=True)
    base_hash = frozen_digest(model)
    initial = {n: v.detach().clone() for n, v in model.named_parameters() if v.requires_grad}
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=1e-4)
    model.train()
    losses = model.loss(**{'batch_inputs': batch['inputs'], 'batch_data_samples': batch['data_samples']})
    loss, log_vars = model.parse_losses(losses)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {n: dict(finite=bool(torch.isfinite(v.grad).all()), nonzero=bool(v.grad.abs().sum() > 0))
                 for n, v in model.named_parameters() if v.requires_grad and v.grad is not None}
    assert len(gradients) == 48 and all(v['finite'] for v in gradients.values())
    assert any(v['nonzero'] for v in gradients.values())
    optimizer.step()
    assert frozen_digest(model) == base_hash
    assert any(not torch.equal(v.detach(), initial[n]) for n, v in model.named_parameters() if v.requires_grad)
    model.eval()
    with torch.no_grad():
        prediction = model.predict(batch['inputs'], batch['data_samples'])[0].pred_instances
    a.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = a.out.with_suffix('.pth')
    torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict()), checkpoint_path)
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=True)
    with torch.no_grad():
        reloaded = model.predict(batch['inputs'], batch['data_samples'])[0].pred_instances
    assert all(torch.equal(getattr(prediction, k), getattr(reloaded, k)) for k in ['bboxes', 'scores', 'labels'])
    reference_loaded = False
    if a.reference_checkpoint:
        reference = torch.load(a.reference_checkpoint, map_location='cpu')
        # num_queries affects only this learned embedding; all other trained state is strict.
        model.num_queries = 900
        model.query_embedding = torch.nn.Embedding(900, model.embed_dims)
        model.query_embedding.requires_grad_(False)
        model.load_state_dict(reference['state_dict'], strict=True)
        reference_loaded = True
        del reference
    write_json(a.out, dict(passed=True, engineering_only=True, device='cpu',
        cuda_initialized=torch.cuda.is_initialized(), actual_image_shape=list(batch['inputs'].shape),
        engineering_num_queries=20, formal_num_queries=900,
        trainable_parameters=params['trainable'], frozen_parameters=params['frozen'],
        lora_layers=len(model.lora_paths), losses={k: float(v) for k, v in log_vars.items()},
        gradient_tensors=len(gradients), frozen_unchanged=True, adapter_updated=True,
        strict_reload=True, predictions_exact=True, reference_checkpoint_strict_load=reference_loaded,
        source='official MMDetection v3.3.0 + project-local LoRALinear',
        checkpoint_sha256=sha256(checkpoint_path), elapsed_seconds=time.perf_counter() - started,
        note='Reduced input and query count are CPU engineering checks; no GPU throughput or formal AP.'))
    checkpoint_path.unlink()
    print(a.out.read_text())


if __name__ == '__main__':
    main()
