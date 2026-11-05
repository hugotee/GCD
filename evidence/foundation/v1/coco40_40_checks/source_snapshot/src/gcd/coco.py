"""COCO2017 bbox projections with sparse category IDs and explicit phase rules."""
import json
from pathlib import Path
from gcd.protocol import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]


def categories():
    reference = json.loads((ROOT/'configs/datasets/coco80_reference.json').read_text())
    return reference['categories']


def bbox_projection(data):
    expected = {c['id']:c['name'] for c in categories()}
    if {c['id']:c['name'] for c in data['categories']} != expected:
        raise ValueError('COCO category IDs/names differ from the frozen 80-class reference')
    if len({x['id'] for x in data['images']}) != len(data['images']):
        raise ValueError('Duplicate COCO image IDs')
    if len({x['id'] for x in data['annotations']}) != len(data['annotations']):
        raise ValueError('Duplicate COCO annotation IDs')
    image_ids = {x['id'] for x in data['images']}
    if any(a['image_id'] not in image_ids or a['category_id'] not in expected for a in data['annotations']):
        raise ValueError('Annotation references an unknown image/category')
    return dict(images=[{k:x[k] for k in ['id','file_name','width','height']} for x in data['images']],
        annotations=[{k:x[k] for k in ['id','image_id','category_id','bbox','area','iscrowd','ignore'] if k in x}
                     for x in data['annotations']],
        categories=sorted(data['categories'],key=lambda x:x['id']))


def phase_data(data, selected_ids, minimum_image_size=32):
    selected_ids=set(selected_ids)
    annotations=[a for a in data['annotations'] if a['category_id'] in selected_ids]
    image_ids={a['image_id'] for a in annotations}
    images=[i for i in data['images'] if i['id'] in image_ids and min(i['width'],i['height'])>=minimum_image_size]
    image_ids={i['id'] for i in images}
    return dict(images=images,annotations=[a for a in annotations if a['image_id'] in image_ids],
                categories=data['categories'])


def verify_images(data, image_root):
    missing=[i['file_name'] for i in data['images'] if not (Path(image_root)/i['file_name']).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} images missing in {image_root}; first={missing[:3]}')


def prepare_coco(coco_root, out, num_old=70, evaluation_only=False, standard=True):
    coco_root,out=Path(coco_root).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Refuse to overwrite a prepared COCO protocol')
    if num_old not in [40,70]:raise ValueError('Supported COCO protocols are 40+40 and 70+10')
    reference=categories();names=[c['name'] for c in reference];ids=[c['id'] for c in reference]
    val_source=coco_root/'annotations/instances_val2017.json'
    val=bbox_projection(json.loads(val_source.read_text()))
    if standard and len(val['images'])!=5000:raise ValueError('Complete COCO2017 val requires 5000 images')
    verify_images(val,coco_root/'val2017')
    train=None
    if not evaluation_only:
        train_source=coco_root/'annotations/instances_train2017.json'
        train=bbox_projection(json.loads(train_source.read_text()))
        if standard and len(train['images'])!=118287:raise ValueError('Complete COCO2017 train requires 118287 images')
        verify_images(train,coco_root/'train2017')
        if {i['id'] for i in train['images']} & {i['id'] for i in val['images']}:
            raise ValueError('Train/validation image-ID overlap')
    out.mkdir(parents=True)
    val_path=out/'val_full80_bbox.json';write_json(val_path,val)
    manifest=dict(passed=True,protocol_frozen=True,dataset='coco2017',engineering_only=not standard,
        protocol='COCO2017_full80_eval_v1' if evaluation_only else f'COCO2017_id_order{num_old}+{80-num_old}_current_positive_v1',
        evaluation_only=evaluation_only,classes=names,category_ids=ids,num_old_classes=None if evaluation_only else num_old,
        coco_root=str(coco_root),phases={},replay_buffer=False,
        source_annotation_sha256=dict(val2017=sha256(val_source)),
        test=dict(ann=str(val_path),sha256=sha256(val_path),image_root=str(coco_root/'val2017'),images=len(val['images'])),
        bbox_projection='Preserve original ID/bbox/area/iscrowd/ignore; omit segmentation and unused metadata.')
    if train is not None:
        manifest['source_annotation_sha256']['train2017']=sha256(coco_root/'annotations/instances_train2017.json')
        phases={name:phase_data(train,selected) for name,selected in [('task1',ids[:num_old]),('task2',ids[num_old:])]}
        for phase,data in phases.items():
            ann=out/f'{phase}_train.json';write_json(ann,data)
            sample_ids={i['id'] for i in data['images'][:4]}
            smoke=dict(data,images=data['images'][:4],annotations=[a for a in data['annotations'] if a['image_id'] in sample_ids])
            smoke_path=out/f'{phase}_smoke.json';write_json(smoke_path,smoke)
            manifest['phases'][phase]=dict(train_ann=str(ann),train_sha256=sha256(ann),
                images=len(data['images']),annotations=len(data['annotations']),
                crowd_annotations=sum(bool(a.get('iscrowd')) for a in data['annotations']),
                image_root=str(coco_root/'train2017'),classes=names[:num_old] if phase=='task1' else names,
                expected_coco_dataset_label_ids=list(range(num_old) if phase=='task1' else range(num_old,80)),
                smoke_eval_ann=str(smoke_path),minimum_image_size=32,crowd_handling='Retain and use native ignore_flag')
        t1={i['id'] for i in phases['task1']['images']};t2={i['id'] for i in phases['task2']['images']}
        manifest.update(stage_image_overlap=len(t1&t2),image_disjoint=not bool(t1&t2),
            hidden_old_noncrowd_instances_in_task2=sum(a['category_id'] in set(ids[:num_old]) and not a.get('iscrowd',0) and a['image_id'] in t2 for a in train['annotations']))
    write_json(out/'manifest.json',manifest)
    return manifest
