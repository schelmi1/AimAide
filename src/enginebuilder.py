import pickle
from collections import OrderedDict

from ultralytics import YOLO
from .utils_trt import EngineBuilder

#https://github.com/triple-Mu/YOLOv8-TensorRT


def build_engine(weights, input_shape, output_pkl):
    model = YOLO(weights)
    model.model.fuse()
    YOLOv8 = model.model.model

    strides = YOLOv8[-1].stride.detach().cpu().numpy()
    reg_max = YOLOv8[-1].dfl.conv.weight.shape[1]


    state_dict = OrderedDict(GD=model.model.yaml['depth_multiple'],
                            GW=model.model.yaml['width_multiple'],
                            strides=strides,
                            reg_max=reg_max)


    for name, value in YOLOv8.state_dict().items():
        value = value.detach().cpu().numpy()
        i = int(name.split('.')[0])
        layer = YOLOv8[i]
        module_name = layer.type.split('.')[-1]
        stem = module_name + '.' + name
        state_dict[stem] = value

    with open(output_pkl, 'wb') as f:
        pickle.dump(state_dict, f)

    builder = EngineBuilder(output_pkl, 'cuda:0')
    builder.build(fp16=True, input_shape=input_shape,iou_thres=.5, conf_thres=.5,topk=10)