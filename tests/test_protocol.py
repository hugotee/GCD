import tempfile
import unittest
from pathlib import Path

from gcd.protocol import parse_xml, select_phase, CLASSES


class ProtocolTests(unittest.TestCase):
    def test_standard_order_and_global_task2_labels(self):
        self.assertEqual(CLASSES[:2], ('aeroplane', 'bicycle'))
        self.assertEqual(CLASSES[15:], ('pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'))

    def test_current_positive_shared_images_and_hidden_annotations(self):
        data = dict(images=[dict(id=1), dict(id=2), dict(id=3)], categories=[], annotations=[
            dict(id=1, image_id=1, category_id=1, iscrowd=0),
            dict(id=2, image_id=1, category_id=16, iscrowd=0),
            dict(id=3, image_id=2, category_id=16, iscrowd=1),
            dict(id=4, image_id=3, category_id=1, iscrowd=0)])
        old = select_phase(data, range(1, 16))
        new = select_phase(data, range(16, 21))
        self.assertEqual([i['id'] for i in old['images']], [1, 3])
        self.assertEqual([i['id'] for i in new['images']], [1])
        self.assertEqual([a['category_id'] for a in new['annotations']], [16])
        self.assertEqual(len(data['annotations']), 4)

    def test_voc_inclusive_pixel_coordinates_and_difficult(self):
        xml = '<annotation><size><width>10</width><height>20</height></size>' \
              '<object><name>person</name><difficult>1</difficult><bndbox>' \
              '<xmin>1</xmin><ymin>2</ymin><xmax>1</xmax><ymax>3</ymax>' \
              '</bndbox></object></annotation>'
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'a.xml'
            p.write_text(xml)
            width, height, anns = parse_xml(p)
        self.assertEqual((width, height), (10, 20))
        self.assertEqual(anns[0]['bbox'], [0, 1, 1, 2])
        self.assertEqual(anns[0]['voc_difficult'], 1)
