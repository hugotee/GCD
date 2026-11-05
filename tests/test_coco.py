import copy
import json
from pathlib import Path
import tempfile
import unittest
from gcd.coco import categories,bbox_projection,phase_data
from gcd.coco_metrics import evaluate_coco
from gcd.runtime import validate_manifest
from gcd.protocol import sha256

class CocoTests(unittest.TestCase):
    def fixture(self):
        return dict(categories=[dict(id=c['id'],name=c['name']) for c in categories()],
            images=[dict(id=i,file_name=f'{i}.jpg',width=100 if i!=4 else 20,height=100) for i in [1,2,3,4]],
            annotations=[dict(id=i,image_id=im,category_id=cat,bbox=[1,1,5,5],area=17,
                              iscrowd=crowd,segmentation=[]) for i,(im,cat,crowd) in enumerate(
                                  [(1,79,0),(1,80,0),(2,90,1),(3,1,0),(4,80,0)],1)])

    def test_sparse_ids_crowd_and_original_area_are_preserved(self):
        raw=self.fixture();before=copy.deepcopy(raw);data=bbox_projection(raw)
        old_ids=[c['id'] for c in categories()[:70]]
        phase=phase_data(data,[c['id'] for c in categories()[70:]])
        self.assertEqual([i['id'] for i in phase['images']],[1,2])
        self.assertEqual([a['category_id'] for a in phase['annotations']],[80,90])
        self.assertEqual(phase['annotations'][1]['iscrowd'],1)
        self.assertEqual(phase['annotations'][0]['area'],17)
        self.assertNotIn('segmentation',phase['annotations'][0])
        self.assertIn(79,old_ids);self.assertNotIn(80,old_ids)
        self.assertEqual(raw,before)
        self.assertEqual({c['id']:i for i,c in enumerate(phase['categories'])}[90],79)

    def test_category_identity_and_duplicate_annotations_are_rejected(self):
        raw=self.fixture();raw['categories'][-1]['id']=91
        with self.assertRaises(ValueError):bbox_projection(raw)
        raw=self.fixture();raw['annotations'][1]['id']=raw['annotations'][0]['id']
        with self.assertRaises(ValueError):bbox_projection(raw)

    def test_evaluation_only_manifest_cannot_be_used_for_training(self):
        with tempfile.TemporaryDirectory() as folder:
            ann=Path(folder)/'val.json';ann.write_text('{}')
            ref=categories();m=dict(passed=True,protocol_frozen=True,dataset='coco2017',evaluation_only=True,
                classes=[c['name'] for c in ref],category_ids=[c['id'] for c in ref],phases={},
                test=dict(ann=str(ann),sha256=sha256(ann)))
            validate_manifest(m,training=False)
            with self.assertRaises(ValueError):validate_manifest(m,training=True)

    def test_standard_coco_metric_handles_sparse_ids_and_empty_predictions(self):
        gt=dict(images=[dict(id=1,width=100,height=100)],categories=[dict(id=80,name='toaster'),dict(id=90,name='toothbrush')],
            annotations=[dict(id=i,image_id=1,category_id=c,bbox=[i*20,0,10,10],area=100,iscrowd=0) for i,c in enumerate([80,90],1)])
        pred=[dict(image_id=1,category_id=a['category_id'],bbox=a['bbox'],score=.9) for a in gt['annotations']]
        result=evaluate_coco(gt,pred,[80,90])
        self.assertAlmostEqual(result['groups']['all2']['COCO_AP'],100)
        self.assertEqual(evaluate_coco(gt,[],[80,90])['groups']['all2']['COCO_AP'],0)
        pred[0]['category_id']=1
        with self.assertRaises(ValueError):evaluate_coco(gt,pred,[80,90])
