"""Standard COCO bbox metrics with original category IDs retained."""
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco(ground_truth,predictions,category_ids,num_old=None):
    valid_ids=set(category_ids);image_ids={i['id'] for i in ground_truth['images']}
    if any(p['category_id'] not in valid_ids or p['image_id'] not in image_ids for p in predictions):
        raise ValueError('Prediction has unknown raw category/image ID')
    gt=COCO();gt.dataset=dict(ground_truth,info=ground_truth.get('info',{}));gt.createIndex()
    if predictions:
        dt=gt.loadRes(predictions)
    else:
        dt=COCO();dt.dataset=dict(images=ground_truth['images'],categories=ground_truth['categories'],annotations=[]);dt.createIndex()
    evaluator=COCOeval(gt,dt,'bbox');evaluator.params.imgIds=sorted(image_ids);evaluator.params.catIds=category_ids
    evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    precision=evaluator.eval['precision'];rows=[]
    for index,cat in enumerate(evaluator.params.catIds):
        values=precision[:,:,index,0,-1]
        def mean(a):return float(a[a>=0].mean()*100) if (a>=0).any() else None
        rows.append(dict(category_id=int(cat),class_index=category_ids.index(cat),category=gt.cats[cat]['name'],
            COCO_AP=mean(values),COCO_AP50=mean(values[0]),COCO_AP75=mean(values[5])))
    groups={}
    for name,selected in [(f'old{num_old}',rows[:num_old]),(f'new{len(rows)-num_old}',rows[num_old:])] if num_old else []:
        groups[name]={m:float(np.mean([r[m] for r in selected if r[m] is not None])) for m in ['COCO_AP','COCO_AP50','COCO_AP75']}
    groups[f'all{len(rows)}']={m:float(np.mean([r[m] for r in rows if r[m] is not None])) for m in ['COCO_AP','COCO_AP50','COCO_AP75']}
    return dict(groups=groups,per_class=rows,official_stats={name:float(v*100) for name,v in zip(
        ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large'],evaluator.stats)})
