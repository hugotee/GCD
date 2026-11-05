import importlib.util
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec=importlib.util.spec_from_file_location('coco_download',Path(__file__).resolve().parents[1]/'scripts/download_coco2017.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class DownloadTests(unittest.TestCase):
    def test_existing_archive_preserves_transport_and_writes_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'.downloads').mkdir()
            archive=root/'.downloads/annotations_trainval2017.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for split in ['train','val']:
                    z.writestr(f'annotations/instances_{split}2017.json','{}')
            digest=hashlib.md5(archive.read_bytes()).hexdigest()
            class Response:
                headers={'Content-Length':str(archive.stat().st_size),'ETag':'fixture'}
                def __enter__(self):return self
                def __exit__(self,*args):pass
            with patch.object(module.urllib.request,'urlopen',return_value=Response()),patch.dict(module.CHECKSUMS,{archive.name:digest}):
                result=module.download(root,archive.name,'unused',archive.stat().st_size,1,'hf-mirror')
            self.assertEqual(result['transport'],'hf-mirror')
            completed=json.loads((root/'.downloads'/f'{archive.name}.complete.json').read_text())
            self.assertTrue(completed['published_md5_verified'])
            self.assertEqual(completed['extracted_files'],2)

    def test_selected_detection_annotations_exclude_unrelated_payloads(self):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as archive:
            archive.writestr('annotations/instances_val2017.json','{}')
            archive.writestr('annotations/captions_val2017.json','{}')
        with zipfile.ZipFile(data) as archive:
            selected=module.safe_members(archive,lambda p:p=='annotations/instances_val2017.json')
            self.assertEqual([i.filename for i in selected],['annotations/instances_val2017.json'])

    def test_path_traversal_is_rejected(self):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as archive:archive.writestr('../outside','bad')
        with zipfile.ZipFile(data) as archive:
            with self.assertRaises(ValueError):module.safe_members(archive,lambda p:True)
