import torchvision.models.detection as td


def get_model(num_class):
    anchor_sizes = ((64, ), (128, ), (256, ), (512, ), (1024, ))
    aspect_ratios = ((0.5, 1.0, 2.0), ) * len(anchor_sizes)
    rpn_anchor_generator = td.anchor_utils.AnchorGenerator(
        anchor_sizes, aspect_ratios)
    # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
    # 使用 imagenet 预训练的网络
    # 返回的是 faster rcnn，网络会再次对 图像数据进行重定义尺寸
    # https://github.com/pytorch/vision/blob/c2ab0c59f42babf9ad01aa616cd8a901daac86dd/torchvision/models/detection/transform.py#L64
    detector = td.fasterrcnn_resnet50_fpn(
        rpn_anchor_generator=rpn_anchor_generator, pretrained=True)
    num_classes = num_class
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = td.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)

    return detector