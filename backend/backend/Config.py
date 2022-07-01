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
    sort_color_map = [0xd9,0xa1,0x46, 0x52,0x5e,0x71,0x2e,0xc4,0xb6, 0x34,0x3a,0x40,0x73,0xa9,0x42, 0x90,0xe0,0xef]
    detection_label_list = ['飞机', '船舶', '储罐', '棒球场', '网球场', '篮球场', '地面田径场', '港口', '桥梁', '大型车辆', '小型车辆', '直升机', '环岛', '足球场', '游泳池', '集装箱起重机']
    detection_color_list = [0xf9,0x41,0x44,0xf3,0x72,0x2c,0xf8,0x96,0x1e,0xf9,
                            0x84,0x4a,0xf9,0xc7,0x4f,0x90,0xbe,0x6d, 0x43,0xaa,0x8b,
                            0x4d,0x90,0x8e, 0x57,0x75,0x90,0x27,0x7d,0xa1,0x0d,0x30,
                            0x82,0x88,0xda,0xe7,0x76,0xcd,0x65,0xff,0xc2,0x47,
                            0xff,0x81,0x33, 0xeb,0x51,0x33]

    '''
    ['#f94144', '#f3722c', '#f8961e', '#f9844a', '#f9c74f', '#90be6d', '#43aa8b',
     '#4d908e', '#577590', '#277da1', '#0d3082', '#88dae7', '#76cd65', '#ffc247',
     '#ff8133', '#eb5133']
    '''
    enable_gpu = False

    