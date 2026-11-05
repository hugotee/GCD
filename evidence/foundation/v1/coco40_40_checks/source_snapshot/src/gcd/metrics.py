"""Distinct VOC07 11-point AP50 and COCO AP metrics, in percent."""
import numpy as np


def voc07_class_ap(ground_truth, predictions, category_id):
    records = {}
    for image in ground_truth['images']:
        records[image['id']] = dict(boxes=[], difficult=[], detected=[])
    positives = 0
    for ann in ground_truth['annotations']:
        if ann['category_id'] != category_id:
            continue
        row = records[ann['image_id']]
        x, y, w, h = ann['bbox']
        row['boxes'].append([x + 1, y + 1, x + w, y + h])
        difficult = bool(ann.get('voc_difficult', ann.get('iscrowd', 0)))
        row['difficult'].append(difficult)
        row['detected'].append(False)
        positives += not difficult
    if positives == 0:
        return None
    detections = [p for p in predictions if p['category_id'] == category_id]
    order = np.argsort(-np.asarray([p['score'] for p in detections]), kind='stable')
    tp, fp = np.zeros(len(order)), np.zeros(len(order))
    for j, idx in enumerate(order):
        pred = detections[idx]
        if pred['image_id'] not in records:
            raise ValueError('Prediction contains image outside the evaluation split')
        row = records[pred['image_id']]
        if not row['boxes']:
            fp[j] = 1
            continue
        boxes = np.asarray(row['boxes'])
        x, y, w, h = pred['bbox']
        box = np.asarray([x + 1, y + 1, x + w, y + h])
        intersection = np.maximum(np.minimum(boxes[:, 2:], box[2:]) -
                                  np.maximum(boxes[:, :2], box[:2]) + 1, 0).prod(axis=1)
        areas = np.maximum(boxes[:, 2:] - boxes[:, :2] + 1, 0).prod(axis=1)
        iou = intersection / np.maximum(w * h + areas - intersection, np.finfo(float).eps)
        best = int(iou.argmax())
        if iou[best] > .5:  # VOC devkit uses strict greater-than.
            if not row['difficult'][best]:
                if row['detected'][best]:
                    fp[j] = 1
                else:
                    tp[j] = 1
                    row['detected'][best] = True
        else:
            fp[j] = 1
    tp, fp = tp.cumsum(), fp.cumsum()
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, np.finfo(float).eps)
    return 100 * sum(float(precision[recall >= r].max()) if (recall >= r).any() else 0.
                     for r in np.linspace(0, 1, 11)) / 11


def evaluate(ground_truth, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    gt.dataset = ground_truth
    gt.createIndex()
    valid_ids = set(gt.imgs)
    for p in predictions:
        if (p['image_id'] not in valid_ids or p['category_id'] not in gt.cats or
                not np.isfinite(p['score']) or not np.isfinite(p['bbox']).all() or
                p['bbox'][2] < 0 or p['bbox'][3] < 0):
            raise ValueError(f'Invalid detection: {p}')
    if predictions:
        dt = gt.loadRes(predictions)
    else:
        dt = COCO()
        dt.dataset = dict(images=ground_truth['images'], categories=ground_truth['categories'], annotations=[])
        dt.createIndex()
    ev = COCOeval(gt, dt, 'bbox')
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    precision = ev.eval['precision']
    rows = []
    for k, cat in enumerate(ev.params.catIds):
        cat = int(cat)
        values = precision[:, :, k, 0, -1]
        v50 = values[0]
        rows.append(dict(category_id=cat, category=gt.cats[cat]['name'],
            group='old15' if cat <= 15 else 'new5',
            VOC07_AP50=voc07_class_ap(ground_truth, predictions, cat),
            COCO_AP=float(values[values >= 0].mean() * 100) if (values >= 0).any() else None,
            COCO_AP50=float(v50[v50 >= 0].mean() * 100) if (v50 >= 0).any() else None))
    groups = {}
    for name in ('old15', 'new5', 'all20'):
        selected = [r for r in rows if name == 'all20' or r['group'] == name]
        groups[name] = {m: float(np.mean([r[m] for r in selected if r[m] is not None]))
                        for m in ('VOC07_AP50', 'COCO_AP50', 'COCO_AP')}
    return dict(groups=groups, per_class=rows)
