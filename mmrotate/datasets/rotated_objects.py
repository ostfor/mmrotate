import os
import pandas as pd
import yaml
from PIL import Image

from .builder import ROTATED_DATASETS
from .dota import DOTADataset


@ROTATED_DATASETS.register_module()
class RotObjectsDataset(DOTADataset):
    """
    RotObjectsDataset inheriting from DOTADataset.

    Parameters
    ----------
    ann_file : str
        Path to the annotation file.
    pipeline : list[dict]
        Processing pipeline.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    CLASSES : tuple
        Tuple of class names in the dataset.

    Dataset Format
    --------------
    dataset_root/
    ├── images/
    │   ├── image0.jpg
    │   └── image1.jpg
    └── annotations.yaml

    Annotation Format
    -----------------
    annotations.yaml:
    - filename: image0.jpg
      ann:
        bboxes:
        - [x1, y1, w1, h1, a1]
        - [x2, y2, w2, h2, a2]
        labels: [0, 1]

    As a Table
    ----------
    |  image_name |  x1 |  y1 | w1  |  h1 |  a1 |  x2 |  y2 | w2  |  h2 |  a2 | labels |
    |-------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|--------|
    | image0.jpg  | x1  | y1  | w1  | h1  | a1  | x2  | y2  | w2  | h2  | a2  | [0, 1] |
    | image1.jpg  | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |  ...   |
    """

    CLASSES = ('object',)
    PALETTE = [(0, 255, 0)]

    def load_annotations(self, ann_file):
        """
        Load annotations from the YAML file.

        Parameters
        ----------
        ann_file : str
            Path to the annotation file.

        Returns
        -------
        list
            List of dictionaries containing the image information and annotations.

        """
        # Load the annotations from the YAML file
        with open(ann_file, 'r') as f:
            annotations = yaml.safe_load(f)

        data_infos = []
        for ann_info in annotations:
            data_info = {}
            img_name = ann_info['filename']
            data_info['filename'] = img_name

            # Retrieve the bounding box coordinates and angles
            bboxes = ann_info['ann'].get('bboxes', [])
            labels = ann_info['ann'].get('labels', [])

            # Store the bounding box information in the annotation dictionary
            data_info['ann'] = {}
            data_info['ann']['bboxes'] = bboxes
            data_info['ann']['labels'] = labels

            data_infos.append(data_info)

        return data_infos

    def load_img(self, img_info):
        """
        Load image from file.

        Parameters
        ----------
        img_info : dict
            Image information.

        Returns
        -------
        PIL.Image
            Loaded image.

        """
        # Load the image using PIL
        img_path = os.path.join(self.img_prefix, img_info['filename'])
        img = Image.open(img_path).convert('RGB')

        # Return the loaded image
       
