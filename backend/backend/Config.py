# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2022//04//03
"""
class Config:
    retrieval_model_path = "params/deploy_road_r/model.pdmodel"
    retrieval_param_path = "params/deploy_road_r/model.pdiparams"

    detection_model_path = "params/retrieval.pdmodel"
    detection_param_path = "params/retrieval.pdiparams"

    sort_model_path = "params/deploy_sort/model.pdmodel"
    sort_param_path = "params/deploy_sort/model.pdiparams"

    constrast_model_path = "params/retrieval.pdmodel"
    constrast_param_path = "params/retrieval.pdiparams"

    sort_category = ["Bare soil","Building","Pavement","Road","Vegetation","Water"]

    enable_gpu = False

    