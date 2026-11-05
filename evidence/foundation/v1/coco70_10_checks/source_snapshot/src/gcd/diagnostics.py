"""Read-only GT-anchored diagnostics; these candidate bins are not TIDE dAP."""
from collections import Counter

import numpy as np


def iou_xywh(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    intersection = np.maximum(np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(axis=2)
    union = a[:, 2:].prod(axis=1)[:, None] + b[:, 2:].prod(axis=1)[None, :] - intersection
    return intersection / np.maximum(union, np.finfo(float).eps)


def candidate_rows(annotations, predictions, threshold=.3):
    """One row per non-difficult GT. Coverage may share queries: not official recall."""
    gt = [a for a in annotations if not a.get('voc_difficult', a.get('iscrowd', 0))]
    ious = iou_xywh([p['bbox'] for p in predictions], [a['bbox'] for a in gt])
    scores = np.asarray([p['score'] for p in predictions])
    labels = np.asarray([p['category_id'] for p in predictions])
    rows = []
    for j, ann in enumerate(gt):
        same = labels == ann['category_id']
        localized = ious[:, j] > .5
        strong = scores >= threshold
        matching = same & localized
        if (matching & strong).any():
            kind = 'same_class_localized'
        elif (localized & strong).any():
            kind = 'other_class_localized'
        elif (same & strong & (ious[:, j] > .1)).any():
            kind = 'same_class_partial_overlap'
        elif (strong & (ious[:, j] > .1)).any():
            kind = 'other_partial_overlap'
        else:
            kind = 'no_candidate_overlap'
        eligible = np.flatnonzero(localized & strong)
        best = int(eligible[np.argmax(scores[eligible])]) if len(eligible) else None
        rows.append(dict(annotation_id=int(ann['id']), category_id=int(ann['category_id']),
            candidate_bin=kind,
            best_same_class_score_iou50=float(scores[matching].max()) if matching.any() else None,
            any_class_iou50_at_threshold=bool((localized & strong).any()),
            same_class_iou50_at_threshold=bool((matching & strong).any()),
            max_cached_iou=float(ious[:, j].max()) if len(predictions) else 0.,
            strongest_localized_label=int(labels[best]) if best is not None else None,
            strongest_localized_score=float(scores[best]) if best is not None else None))
    return rows


def prediction_bins(annotations, predictions, threshold=.3):
    """Score-ordered VOC-style TP assignment plus descriptive FP subtypes."""
    counts, confusion = Counter(), Counter()
    used = set()
    labels = np.asarray([a['category_id'] for a in annotations])
    difficult = np.asarray([bool(a.get('voc_difficult', a.get('iscrowd', 0))) for a in annotations], dtype=bool)
    candidates = sorted((p for p in predictions if p['score'] >= threshold), key=lambda p: -p['score'])
    ious = iou_xywh([p['bbox'] for p in candidates], [a['bbox'] for a in annotations])
    for i, pred in enumerate(candidates):
        prefix = 'old' if pred['category_id'] <= 15 else 'new'
        same = np.flatnonzero(labels == pred['category_id'])
        best = int(same[np.argmax(ious[i, same])]) if len(same) else None
        if best is not None and ious[i, best] > .5:
            if difficult[best]:
                counts[prefix + ':ignored_difficult'] += 1
            elif best in used:
                counts[prefix + ':duplicate'] += 1
            else:
                counts[prefix + ':true_positive'] += 1
                used.add(best)
            continue
        other = np.flatnonzero((labels != pred['category_id']) & ~difficult & (ious[i] > .5))
        if len(other):
            matched = int(other[np.argmax(ious[i, other])])
            true = int(labels[matched])
            counts[prefix + ':wrong_class'] += 1
            confusion[(true, int(pred['category_id']))] += 1
        elif len(same) and float(ious[i, same].max()) > .1:
            counts[prefix + ':localization'] += 1
        elif len(annotations) and float(ious[i].max()) > .1:
            counts[prefix + ':other_overlap'] += 1
        else:
            counts[prefix + ':background'] += 1
    return counts, confusion


def image_stratum(annotations):
    cats = {a['category_id'] for a in annotations if not a.get('voc_difficult', a.get('iscrowd', 0))}
    old, new = any(c <= 15 for c in cats), any(c > 15 for c in cats)
    return 'old_and_new' if old and new else 'old_only' if old else 'new_only' if new else 'no_nondifficult_gt'


def choose_probe(data, count=128, seed=20261104):
    """Preselect balanced trainval images with old GT, without model scores/test data."""
    grouped = {}
    for ann in data['annotations']:
        grouped.setdefault(ann['image_id'], []).append(ann)
    pools = {'old_only': [], 'old_and_new': []}
    for image in data['images']:
        key = image_stratum(grouped.get(image['id'], []))
        if key in pools:
            pools[key].append(image['id'])
    rng = np.random.default_rng(seed)
    ids = []
    for key, pool in pools.items():
        n = count // 2
        if len(pool) < n:
            raise ValueError(f'Insufficient probe pool {key}')
        ids.extend(int(x) for x in rng.choice(sorted(pool), n, replace=False))
    return sorted(ids)


def representation_changes(reference, candidate):
    """Separate corresponding-vector direction, norm and within-text topology."""
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError('Expected matching [images, classes, feature] arrays')
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    if (na < 1e-12).any() or (nb < 1e-12).any():
        raise ValueError('Cosine is undefined for a zero representation')
    ua, ub = a / na[..., None], b / nb[..., None]
    cosine = (ua * ub).sum(-1)
    ga, gb = ua @ ua.transpose(0, 2, 1), ub @ ub.transpose(0, 2, 1)
    off_diagonal = ~np.eye(a.shape[1], dtype=bool)
    return dict(max_abs_difference=float(np.abs(a - b).max()),
        mean_corresponding_cosine=float(cosine.mean()),
        mean_one_minus_cosine=float((1 - cosine).mean()),
        reference_mean_norm=float(na.mean()), candidate_mean_norm=float(nb.mean()),
        mean_norm_ratio=float((nb / na).mean()),
        mean_abs_pairwise_cosine_change=float(np.abs(ga - gb)[:, off_diagonal].mean()))
