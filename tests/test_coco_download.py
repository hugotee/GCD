import importlib.util
import io
from pathlib import Path
import unittest
import zipfile

spec=importlib.util.spec_from_file_location('coco_download',Path(__file__).resolve().parents[1]/'scripts/download_coco2017.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class DownloadTests(unittest.TestCase):
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
