#!/usr/bin/env python3
"""Fixed-checkpoint, no-training 15/20-class prompt and representation probe."""
import argparse
from collections import defaultdict
import copy
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcd.protocol import CLASSES, convert_voc, sha256, write_json
from gcd.runtime import read_runtime, validate_gpus


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpus', default='2')
    p.add_argument('--images', type=int, default=128)
    p.add_argument('--cpu-smoke', action='store_true', help='Small-input/query correctness check only; not the registered probe')
    p.add_argument('--plan', type=Path, help='Pre-registered 128-image plan, required for the full GPU probe')
    a = p.parse_args()
    runtime = read_runtime(a.runtime)
    if validate_gpus(a.gpus, runtime['authorized_gpus']) != [2]:
        raise ValueError('This diagnostic run is scoped to physical GPU2 only')
    if not a.cpu_smoke:
        if a.plan is None:
            p.error('Pass the independently registered --plan for the full GPU probe')
        usage = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'], text=True)
        memory, utilization = map(int, usage.strip().split(','))
        if memory > 512 or utilization > 5:
            raise RuntimeError('GPU2 is busy; no probe started')
    elif a.images != 2:
        raise ValueError('CPU smoke is restricted to two diagnostic images')
    device = 'cpu' if a.cpu_smoke else 'cuda:0'
    queries = 20 if a.cpu_smoke else 900
    size = (96, 128) if a.cpu_smoke else (800, 1333)
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if a.cpu_smoke else '2'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, runtime['mmdet_root'])
    import numpy as np
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.logging import MMLogger
    from mmdet.registry import MODELS, DATASETS
    from mmdet.utils import register_all_modules
    import gcd.model
    from gcd.model import frozen_digest
    from gcd.diagnostics import choose_probe, iou_xywh, image_stratum
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    register_all_modules()
    MMLogger.get_current_instance().setLevel(logging.WARNING)
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(runtime['manifest']).read_text())
    train = convert_voc(manifest['voc_root'], 'trainval')
    assert a.images > 0 and a.images % 2 == 0
    selected = choose_probe(train, count=a.images)
    if not a.cpu_smoke:
        registered = json.loads(a.plan.read_text())
        assert registered['manifest_sha256'] == sha256(runtime['manifest'])
        assert registered['selected_image_ids'] == selected
    selection = set(selected)
    data = dict(train, images=[i for i in train['images'] if i['id'] in selection],
                annotations=[x for x in train['annotations'] if x['image_id'] in selection])
    anns = defaultdict(list)
    for row in data['annotations']:
        anns[row['image_id']].append(row)
    plan = dict(diagnostic_only=True, training_performed=False, split='VOC2007 trainval',
        seed=20261104, selected_image_ids=selected,
        selected_filenames=[i['file_name'] for i in data['images']],
        old_only_images=sum(image_stratum(anns[i]) == 'old_only' for i in selected),
        old_and_new_images=sum(image_stratum(anns[i]) == 'old_and_new' for i in selected),
        prompt_variants=['old15', 'all20'], thresholds=[.05, .3, .5],
        engineering_only=a.cpu_smoke, num_queries=queries, short_edge=size[0], max_long_edge=size[1],
        note='Selection uses train annotations only and is fixed before any probe inference; not formal AP.')
    write_json(a.out / 'selection.json', plan)
    write_json(a.out / 'probe_annotations.json', data)
    cfg = Config.fromfile(str(ROOT / 'configs/model/gdino_t.py')).model
    for key in ['as_two_stage', 'share_pred_layer', 'num_pred_layer', 'train_cfg', 'test_cfg']:
        cfg.bbox_head.pop(key, None)
    cfg.update(type='SingleLoRAGroundingDINO', initial_checkpoint=runtime['pretrained_checkpoint'],
               lora=dict(rank=16, alpha=32, dropout=0), train_label_ids=list(range(20)), expected_classes=list(CLASSES))
    cfg.language_model.name = runtime['bert_path']
    cfg.encoder.num_cp, cfg.backbone.with_cp = 0, False
    assert sha256(runtime['pretrained_checkpoint']) == runtime['pretrained_sha256']
    model = MODELS.build(cfg)
    model.init_weights()
    model.eval().to(device)
    dataset = DATASETS.build(dict(type='CocoDataset', ann_file=str(a.out / 'probe_annotations.json'),
        data_prefix=dict(img=manifest['phases']['task1']['image_root'] + '/'),
        metainfo=dict(classes=CLASSES), return_classes=True, test_mode=True,
        pipeline=[dict(type='LoadImageFromFile'), dict(type='FixScaleResize', scale=size, keep_ratio=True),
                  dict(type='LoadAnnotations', with_bbox=True), dict(type='PackDetInputs',
                      meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'text', 'custom_entities'))]))
    captured = {}
    def encoder_input(module, args, kwargs):
        captured['pre_text'] = kwargs['memory_text'].detach()
    def head_output(module, args, output):
        captured['fused_text'] = args[2].detach()
        captured['queries'] = args[0][-1].detach()
        captured['logits'] = output[0][-1].detach()
        captured['boxes'] = output[1][-1].detach()
    handles = [model.encoder.register_forward_pre_hook(encoder_input, with_kwargs=True),
               model.bbox_head.register_forward_hook(head_output)]
    checkpoints = dict(B0=Path(runtime['pretrained_checkpoint']),
        B1=a.reference_root / 'checkpoints/task1_15classes.pth',
        B2=a.reference_root / 'checkpoints/task2_20classes.pth')
    arrays, all_rows, repeats, provenance = {}, [], {}, {}
    started = time.perf_counter()
    if not a.cpu_smoke:
        torch.cuda.reset_peak_memory_stats()
    for name, checkpoint in checkpoints.items():
        digest = sha256(checkpoint)
        if name != 'B0':
            if a.cpu_smoke:
                model.num_queries = 900
                model.query_embedding = torch.nn.Embedding(900, model.embed_dims).to(device)
                model.query_embedding.requires_grad_(False)
            expected = json.loads((a.reference_root / f'results/{name}/summary.json').read_text())
            assert digest == expected['provenance']['initial_checkpoint_sha256']
            state = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(state['state_dict'], strict=True)
            del state
        if a.cpu_smoke:
            model.num_queries = queries
            model.query_embedding = torch.nn.Embedding.from_pretrained(
                model.query_embedding.weight[:queries].detach().clone(), freeze=True)
        provenance[name] = dict(checkpoint_sha256=digest)
        # No optimizer is constructed. Compare frozen hashes and adapter tensors.
        frozen_before = frozen_digest(model)
        before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        text_pre, text_fused = {'old15': [], 'all20': []}, {'old15': [], 'all20': []}
        for n in range(len(dataset)):
            packed = dataset[n]
            iid = int(packed['data_samples'].img_id)
            batch = model.data_preprocessor(pseudo_collate([packed]), training=False)
            for variant, classes in [('old15', CLASSES[:15]), ('all20', CLASSES)]:
                samples = copy.deepcopy(batch['data_samples'])
                samples[0].set_metainfo(dict(text=classes))
                with torch.inference_mode():
                    model.predict(batch['inputs'], samples)
                mapping = samples[0].token_positive_map
                raw = captured['logits'][0].sigmoid()
                scores = torch.stack([raw[:, mapping[i]] .mean(-1) for i in range(1, len(classes) + 1)], dim=1).cpu().numpy()
                assert scores.shape == (queries, len(classes)) and np.isfinite(scores).all()
                means_pre = torch.stack([captured['pre_text'][0, mapping[i]].mean(0) for i in range(1, 16)]).cpu().numpy()
                means_fused = torch.stack([captured['fused_text'][0, mapping[i]].mean(0) for i in range(1, 16)]).cpu().numpy()
                text_pre[variant].append(means_pre)
                text_fused[variant].append(means_fused)
                boxes = captured['boxes'][0].cpu().numpy()
                w, h = packed['data_samples'].ori_shape[1], packed['data_samples'].ori_shape[0]
                left_top = np.maximum((boxes[:, :2] - boxes[:, 2:] / 2) * [w, h], 0)
                right_bottom = np.minimum((boxes[:, :2] + boxes[:, 2:] / 2) * [w, h], [w, h])
                xywh = np.concatenate([left_top, np.maximum(right_bottom - left_top, 0)], axis=1)
                objects = [x for x in anns[iid] if x['category_id'] <= 15 and not x['iscrowd']]
                ious = iou_xywh(xywh, [x['bbox'] for x in objects])
                query_vectors = captured['queries'][0].cpu().numpy()
                for j, obj in enumerate(objects):
                    cat = obj['category_id'] - 1
                    localized = ious[:, j] > .5
                    best = int(ious[:, j].argmax())
                    q, t = query_vectors[best], means_fused[cat]
                    row = dict(model=name, prompt=variant, image_id=iid, annotation_id=int(obj['id']),
                        category_id=cat + 1, stratum=image_stratum(anns[iid]),
                        max_raw_query_iou=float(ious[best, j]), any_raw_query_iou50=bool(localized.any()),
                        best_true_class_score_iou50=float(scores[localized, cat].max()) if localized.any() else None,
                        best_iou_query_true_score=float(scores[best, cat]),
                        best_iou_query_winning_category=int(scores[best].argmax()) + 1,
                        query_norm=float(np.linalg.norm(q)), text_norm=float(np.linalg.norm(t)),
                        best_iou_query_text_cosine=float(q @ t / max(np.linalg.norm(q) * np.linalg.norm(t), 1e-12)))
                    for threshold in [.05, .3, .5]:
                        row[f'true_score_ge_{threshold}'] = bool((scores[localized, cat] >= threshold).any()) if localized.any() else False
                    all_rows.append(row)
                if n == 0 and variant == 'all20':
                    first = (captured['logits'].clone(), captured['boxes'].clone())
                    with torch.inference_mode():
                        model.predict(batch['inputs'], samples)
                    repeats[name] = dict(logit_max_abs_difference=float((first[0][torch.isfinite(first[0])] - captured['logits'][torch.isfinite(first[0])]).abs().max()),
                                         box_max_abs_difference=float((first[1] - captured['boxes']).abs().max()))
            if (n + 1) % 16 == 0:
                write_json(a.out / 'progress.json', dict(model=name, images=n + 1,
                    elapsed_seconds=time.perf_counter() - started))
                print(name, n + 1, flush=True)
        assert all(torch.equal(p.detach().cpu(), before[n]) for n, p in model.named_parameters() if p.requires_grad)
        assert frozen_digest(model) == frozen_before
        for variant in ['old15', 'all20']:
            arrays[f'{name}_{variant}_pre'] = np.stack(text_pre[variant])
            arrays[f'{name}_{variant}_fused'] = np.stack(text_fused[variant])
    for h in handles:
        h.remove()
    if any(max(v.values()) > 1e-5 for v in repeats.values()):
        raise RuntimeError('Repeat-forward numerical difference exceeds the registered tolerance')
    np.savez_compressed(a.out / 'text_features.npz', **arrays)
    write_json(a.out / 'object_scores.json', all_rows)
    write_json(a.out / 'completion.json', dict(passed=True, diagnostic_only=True, engineering_only=a.cpu_smoke,
        training_performed=False, physical_gpu=None if a.cpu_smoke else 2,
        cuda_initialized=torch.cuda.is_initialized(), images=len(dataset), forwards=6 * len(dataset) + 3,
        num_queries=queries, checkpoint_provenance=provenance, repeat_checks=repeats,
        adapter_parameters_unchanged=True, frozen_parameters_unchanged=True,
        text_feature_sha256=sha256(a.out / 'text_features.npz'), object_scores_sha256=sha256(a.out / 'object_scores.json'),
        selection_sha256=sha256(a.out / 'selection.json'),
        peak_allocated_gib=None if a.cpu_smoke else torch.cuda.max_memory_allocated() / 2**30,
        elapsed_seconds=time.perf_counter() - started,
        note=('Small-input CPU smoke with 20 queries; validates hooks only, not mechanism evidence.' if a.cpu_smoke else
              'Train-side paired diagnostic; full raw900 queries and shared old15 score columns; not full-test AP.')))


if __name__ == '__main__':
    main()
