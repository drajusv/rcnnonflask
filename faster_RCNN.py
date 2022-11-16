from collections import OrderedDict
from itertools import chain
from typing import List, Optional, Tuple


import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from backbone_resnet import (
    BackboneWithFPN,
    ResNetBackbones,
    get_resnet_backbone,
    get_resnet_fpn_backbone,
)

#from pytorch_faster_rcnn_tutorial.utils import from_dict_to_boundingbox


def get_anchor_generator(
    anchor_size: Optional[Tuple[Tuple[int]]] = None,
    aspect_ratios: Optional[Tuple[Tuple[float]]] = None,
) -> AnchorGenerator:
    """Returns the anchor generator."""
    if anchor_size is None:
        anchor_size = ((16,), (32,), (64,), (128,))
    if aspect_ratios is None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_size)

    anchor_generator = AnchorGenerator(sizes=anchor_size, aspect_ratios=aspect_ratios)
    return anchor_generator


def get_roi_pool(
    featmap_names: Optional[List[str]] = None,
    output_size: int = 7,
    sampling_ratio: int = 2,
) -> MultiScaleRoIAlign:
    """Returns the ROI Pooling"""
    if featmap_names is None:
        # default for resnet with FPN
        featmap_names = ["0", "1", "2", "3"]

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=featmap_names,
        output_size=output_size,
        sampling_ratio=sampling_ratio,
    )

    return roi_pooler


def get_faster_rcnn(
    backbone: torch.nn.Module,
    anchor_generator: AnchorGenerator,
    roi_pooler: MultiScaleRoIAlign,
    num_classes: int,
    image_mean: List[float] = [0.485, 0.456, 0.406],
    image_std: List[float] = [0.229, 0.224, 0.225],
    min_size: int = 512,
    max_size: int = 1024,
    **kwargs,
) -> FasterRCNN:
    """Returns the Faster-RCNN model. Default normalization: ImageNet"""
    model = FasterRCNN(
        backbone=backbone,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        num_classes=num_classes,
        image_mean=image_mean,  # ImageNet
        image_std=image_std,  # ImageNet
        min_size=min_size,
        max_size=max_size,
        **kwargs,
    )
    model.num_classes = num_classes
    model.image_mean = image_mean
    model.image_std = image_std
    model.min_size = min_size
    model.max_size = max_size

    return model


def get_faster_rcnn_resnet(
    num_classes: int,
    backbone_name: ResNetBackbones,
    anchor_size: Tuple[Tuple[int, ...], ...],
    aspect_ratios: Tuple[Tuple[float, ...]],
    fpn: bool = True,
    min_size: int = 512,
    max_size: int = 1024,
    **kwargs,
) -> FasterRCNN:
    """
    Returns the Faster-RCNN model with resnet backbone with and without fpn.
    anchor_size can be for example: ((16,), (32,), (64,), (128,))
    aspect_ratios can be for example: ((0.5, 1.0, 2.0),)

    Please note that you specify the aspect ratios for all layers, because we perform:
    aspect_ratios = aspect_ratios * len(anchor_size)

    If you wish more control, change this line accordingly.
    """

    # Backbone
    if fpn:
        backbone: BackboneWithFPN = get_resnet_fpn_backbone(backbone_name=backbone_name)
    else:
        backbone: torch.nn.Sequential = get_resnet_backbone(backbone_name=backbone_name)

    # Anchors
    anchor_size = anchor_size
    aspect_ratios = aspect_ratios * len(anchor_size)
    anchor_generator = get_anchor_generator(
        anchor_size=anchor_size, aspect_ratios=aspect_ratios
    )

    # ROI Pool
    # performing a forward pass to get the number of featuremap names
    # this is required for the get_roi_pool function
    # TODO: there is probably a better way to get the featuremap names (without a forward pass)
    with torch.no_grad():
        backbone.eval()
        random_input = torch.rand(size=(1, 3, 512, 512))
        features = backbone(random_input)

    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    featmap_names = [key for key in features.keys() if key.isnumeric()]

    roi_pool = get_roi_pool(featmap_names=featmap_names)

    # Model
    return get_faster_rcnn(
        backbone=backbone,
        anchor_generator=anchor_generator,
        roi_pooler=roi_pool,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        **kwargs,
    )


