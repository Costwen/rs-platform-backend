# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2022//04//03
"""
class Config:
    retrieval_model_path = "params/deploy_road_r/model.pdmodel"
    retrieval_param_path = "params/deploy_road_r/model.pdiparams"

    detection_model_path = "params/deploy_detection/model.pdmodel"
    detection_param_path = "params/deploy_detection/model.pdiparams"

    sort_model_path = "params/deploy_sort/model.pdmodel"
    sort_param_path = "params/deploy_sort/model.pdiparams"

    constrast_model_path = "params/deploy_contrast/model.pdmodel"
    constrast_param_path = "params/deploy_contrast/model.pdiparams"

    sort_category = ["Bare soil","Building","Pavement","Road","Vegetation","Water"]
    sort_color_map = ['#d9a146', '#525e71', '#2ec4b6', '#343a40', '#73a942', '#90e0ef']

    enable_gpu = False

    