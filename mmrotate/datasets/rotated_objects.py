# Copyright (c) OpenMMLab. All rights reserved.
import os

import pandas as pd
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
    └── annotations.csv

    Annotation Format
    -----------------
    annotations.csv:
    image_name,x,y,width,height,angle
    image0.jpg,10,20,100,200,30
    image1.jpg,50,100,150,250,45

    As a Table
    ----------
    |  image_name |  x  |   y  | width | height | angle |
    |-------------|-----|------|-------|--------|-------|
    | image0.jpg  |  10 |  20  |  100  |  200   |   30  |
    | image1.jpg  |  50 |  100 |  150  |  250   |   45  |
    """
    CLASSES = ('object', )
    PALETTE = [
        (0, 255, 0),
    ]
    def load_annotations(self, ann_file):
        # Load the annotations from the CSV file
        annotations = pd.read_csv(ann_file)

        data_infos = []
        for _, row in annotations.iterrows():
            data_info = {}
            img_name = row['image_name']
            data_info['filename'] = img_name

            # Retrieve the bounding box coordinates and angle
            x, y, w, h, a  = (
                row['x'], row['y'], row['width'], row['height'], row['angle']
            )

            # Store the bounding box information in the annotation dictionary
            data_info['ann'] = {}
            data_info['ann']['bboxes'] = [[x, y, w, h, a]]
            data_info['ann']['labels'] = [0]  # Assuming only one class 'plane'

            data_infos.append(data_info)

        return data_infos

    def load_img(self, img_info):
        # Load the image using PIL
        img_path = os.path.join(self.img_prefix, img_info['filename'])
        img = Image.open(img_path).convert('RGB')

        # Return the loaded image
        return img


