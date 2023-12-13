import os
import importlib

import torch.nn as nn


def get_class(target_class:str, dir_path:str, dir_name:str):
    module_names = [mf[0:-3] for mf in os.listdir(dir_path) if mf.endswith(".py") and not mf.startswith("_")]
    modules = [importlib.import_module("{}.{}".format(dir_name, x)) for x in module_names]
    target_cls = None
    for module in modules:
        try:
            target_cls = getattr(module, target_class)
        except (ImportError, AttributeError) as e:
            continue
    return target_cls


def model_factory(model_name:str="UNet"):
    model_file_dir = __file__.split(os.path.sep)[:-2]
    model_file_dir.append("model")
    model = get_class(model_name, "{}".format(os.path.sep).join(model_file_dir), model_file_dir[-1])
    if model == None:
        raise ValueError("model {} is not found.".format(model_name))
    return model


def dataset_factory(dataset_cls_name:str):
    dataset_file_dir = __file__.split(os.path.sep)[:-2]
    dataset_file_dir.append("dataset")
    Dataset = get_class(dataset_cls_name, "{}".format(os.path.sep).join(dataset_file_dir), dataset_file_dir[-1])
    if Dataset == None:
        raise ValueError("dataset {} is not found.".format(dataset_cls_name))
    return Dataset


def loss_factory(loss_name:str):
    loss_file_dir = __file__.split(os.path.sep)[:-2]
    loss_file_dir.append("loss")
    loss = get_class(loss_name, "{}".format(os.path.sep).join(loss_file_dir), loss_file_dir[-1])
    if loss == None:
        raise ValueError("loss {} is not found.".format(loss_name))
    return loss

if __name__ == "__main__":
    ret = model_factory()
    print(ret)
    
    # print(UNet)
# end main