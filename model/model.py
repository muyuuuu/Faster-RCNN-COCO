import torchvision.models.detection as td


def get_model():
    anchor_sizes = ((64, ), (128, ), (256, ), (512, ), (1024, ))
    aspect_ratios = ((0.5, 1.0, 2.0), ) * len(anchor_sizes)
    rpn_anchor_generator = td.anchor_utils.AnchorGenerator(
        anchor_sizes, aspect_ratios)
    # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
    detector = td.__dict__['fasterrcnn_resnet50_fpn'](
        num_classes=515,
        rpn_anchor_generator=rpn_anchor_generator,
        pretrained=False)

    return detector