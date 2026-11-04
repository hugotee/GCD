"""VOC2007 XML conversion and reproducible current-positive 15+5 splits."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

CLASSES = tuple('aeroplane bicycle bird boat bottle bus car cat chair cow diningtable '
                'dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split())
PROTOCOL = 'VOC07_alphabetical15+5_current_positive_v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def parse_xml(path):
    tree = ET.parse(path).getroot()
    width, height = (int(tree.findtext(f'size/{s}')) for s in ('width', 'height'))
    objects = []
    for obj in tree.findall('object'):
        name = obj.findtext('name').strip().lower()
        if name not in CLASSES:
            raise ValueError(f"Unknown category {name} in {path}")
        x1, y1, x2, y2 = (float(obj.findtext(f'bndbox/{s}'))
                          for s in ('xmin', 'ymin', 'xmax', 'ymax'))
        if not (1 <= x1 <= x2 <= width and 1 <= y1 <= y2 <= height):
            raise ValueError(f"Out-of-image VOC box in {path}: {(x1, y1, x2, y2)}")
        difficult = int(obj.findtext('difficult', '0'))
        objects.append(dict(category_id=CLASSES.index(name) + 1,
                            bbox=[x1 - 1, y1 - 1, x2 - x1 + 1, y2 - y1 + 1],
                            area=(x2 - x1 + 1) * (y2 - y1 + 1),
                            iscrowd=difficult, voc_difficult=difficult))
    return width, height, objects


def convert_voc(voc_root, split):
    root = Path(voc_root)
    ids = (root / 'ImageSets/Main' / f'{split}.txt').read_text().split()
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate split image IDs')
    data = dict(images=[], annotations=[], categories=[dict(id=i + 1, name=n)
                                                      for i, n in enumerate(CLASSES)])
    for name in ids:
        if not (root / 'JPEGImages' / f'{name}.jpg').is_file():
            raise FileNotFoundError(name)
        width, height, objects = parse_xml(root / 'Annotations' / f'{name}.xml')
        image_id = int(name)
        data['images'].append(dict(id=image_id, file_name=f'{name}.jpg', width=width, height=height))
        for obj in objects:
            data['annotations'].append(dict(id=len(data['annotations']) + 1,
                                             image_id=image_id, **obj))
    return data


def select_phase(data, category_ids):
    ids = set(category_ids)
    annotations = [a.copy() for a in data['annotations']
                   if a['category_id'] in ids and not a['iscrowd']]
    image_ids = {a['image_id'] for a in annotations}
    return dict(images=[i.copy() for i in data['images'] if i['id'] in image_ids],
                annotations=annotations, categories=data['categories'])


def prepare(voc_root, out, standard=True):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'Refuse to overwrite protocol: {out}')
    voc = Path(voc_root).resolve()
    train, test = convert_voc(voc, 'trainval'), convert_voc(voc, 'test')
    if {i['id'] for i in train['images']} & {i['id'] for i in test['images']}:
        raise ValueError('Train/test image-ID leakage')
    phases = {k: select_phase(train, cats) for k, cats in
              [('task1', range(1, 16)), ('task2', range(16, 21))]}
    overlap = len({i['id'] for i in phases['task1']['images']} &
                  {i['id'] for i in phases['task2']['images']})
    if standard:
        assert (len(train['images']), len(test['images'])) == (5011, 4952)
        assert [len(phases[p]['images']) for p in ('task1', 'task2')] == [4473, 1006]
        assert overlap == 468
        # Independent source of image membership, not the converted annotations.
        for key, names in [('task1', CLASSES[:15]), ('task2', CLASSES[15:])]:
            positive = set()
            for c in names:
                for line in (voc / 'ImageSets/Main' / f'{c}_trainval.txt').read_text().splitlines():
                    image, flag = line.split()
                    if int(flag) == 1:
                        positive.add(int(image))
            assert positive == {i['id'] for i in phases[key]['images']}
    out.mkdir(parents=True)
    manifest = dict(passed=True, protocol_frozen=True, protocol=PROTOCOL,
                    engineering_only=not standard, classes=list(CLASSES), replay_buffer=False,
                    image_disjoint=False, stage_image_overlap=overlap, phases={},
                    voc_root=str(voc), source_split_hashes={s: sha256(voc / 'ImageSets/Main' / f'{s}.txt')
                                                          for s in ('trainval', 'test')})
    for key, data in phases.items():
        path = out / f'{key}_train.json'
        write_json(path, data)
        smoke = dict(data, images=data['images'][:4])
        smoke_ids = {x['id'] for x in smoke['images']}
        smoke['annotations'] = [a for a in data['annotations'] if a['image_id'] in smoke_ids]
        smoke_path = out / f'{key}_smoke_eval.json'
        write_json(smoke_path, smoke)
        classes = CLASSES[:15] if key == 'task1' else CLASSES
        manifest['phases'][key] = dict(train_ann=str(path), train_sha256=sha256(path),
            images=len(data['images']), annotations=len(data['annotations']),
            image_root=str(voc / 'JPEGImages'), classes=list(classes),
            expected_coco_dataset_label_ids=list(range(15) if key == 'task1' else range(15, 20)),
            smoke_eval_ann=str(smoke_path))
    test_path = out / 'test_full20.json'
    write_json(test_path, test)
    manifest['test'] = dict(ann=str(test_path), sha256=sha256(test_path),
        image_root=str(voc / 'JPEGImages'), images=len(test['images']))
    write_json(out / 'manifest.json', manifest)
    return manifest
