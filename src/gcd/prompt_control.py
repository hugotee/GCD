"""Semantic label and paired-input checks for the bounded Task2 prompt control."""
import hashlib
import json

import numpy as np

from gcd.protocol import CLASSES


def arm_spec(arm):
    if arm not in ('new5', 'all20'):
        raise ValueError('Expected new5 or all20')
    return dict(classes=list(CLASSES[15:] if arm == 'new5' else CLASSES),
                local_label_ids=list(range(5) if arm == 'new5' else range(15, 20)),
                global_label_offset=15 if arm == 'new5' else 0)


def sample_fingerprint(image_id, image, boxes, labels, arm):
    """Ignore prompt spelling/local numbering, but require identical global GT."""
    spec = arm_spec(arm)
    labels = np.asarray(labels, dtype=np.int64)
    if not set(labels.tolist()) <= set(spec['local_label_ids']):
        raise ValueError('Unexpected supervision outside Task2 new classes')
    global_labels = labels + spec['global_label_offset']
    h = hashlib.sha256()
    h.update(str(int(image_id)).encode())
    for a in [np.asarray(image), np.asarray(boxes, dtype=np.float32), global_labels]:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def compare_traces(left, right, expected_images):
    if len(left) != len(right):
        raise ValueError('Arms have different microstep counts')
    if left != right:
        raise ValueError('Arms differ in image order, transformed pixels, or global GT')
    ids = [i for row in left for i in row['image_ids']]
    if len(ids) != expected_images or len(set(ids)) != expected_images:
        raise ValueError('A full epoch must visit each Task2 image exactly once')
    return dict(passed=True, microsteps=len(left), images=len(ids), unique_images=len(set(ids)),
                paired_input_sha256=hashlib.sha256(json.dumps(left, sort_keys=True).encode()).hexdigest())
