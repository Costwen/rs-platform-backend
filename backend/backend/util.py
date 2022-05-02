# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2022//04//03
"""

# import paddle.inference as paddle_infer
import PIL
import time
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import cv2

Map_url_template = "https://webst01.is.autonavi.com/appmaptile?style=6&x={}&y={}&z=18&scl=1"


class Predictor:
    def __init__(self, config):
        self.config = config
        # self.contrast_config = paddle_infer.Config(config.constrast_model_path, config.constrast_param_path)
        # self.contrast_predictor = paddle_infer.create_predictor(self.contrast_config)
        # self.sort_config = paddle_infer.Config(config.sort_model_path, config.sort_param_path)
        # self.sort_predictor = paddle_infer.create_predictor(self.sort_config)

        # self.retrieval_config = paddle_infer.Config(config.retrieval_model_path, config.retrieval_param_path)
        # if config.enable_gpu:
        #     self.retrieval_config.enable_use_gpu(memory_pool_init_size_mb=2048,device_id=0)
        # self.retrieval_predictor = paddle_infer.create_predictor(self.retrieval_config)

        # self.detection_config = paddle_infer.Config(config.detection_model_path, config.detection_param_path)
        # self.detection_predictor = paddle_infer.create_predictor(self.detection_config)

    @classmethod
    def _get_color_map_list(cls,num_classes, custom_color=None):
        num_classes += 1
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = color_map[3:]

        if custom_color:
            color_map[:len(custom_color)] = custom_color
        return color_map

    @classmethod
    def _get_pseudo_color_map(cls, pred, color_map=None):
        pred_mask = PIL.Image.fromarray(pred.astype("uint8"), mode='P')
        if color_map is None:
            color_map = Predictor._get_color_map_list(num_classes=256)
        pred_mask.putpalette(color_map)
        return pred_mask

    @classmethod
    def _normalize(cls, im, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        im = im.astype(np.float32, copy=False) / 255.0
        im -= mean
        im /= std
        return im

    '''
        利用图像闭运算去除结果中的空洞。
    '''
    def morph_close(self, image, size):
        kernel = np.ones((size,size),np.uint8)
        result = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
        return result

    def contrast_predict(self, np_img):
        pass

    def sort_predict(self, np_img):
        pass

    def detection_predict(self, np_img):
        pass

    def retrieval_predict(self, file):
        input_names = self.retrieval_predictor.get_input_names()
        input_handle = self.retrieval_predictor.get_input_handle(input_names[0])
        input_data = np.array(file.resize([1024,1024])).astype("float32")
        input_data = self._normalize(input_data)
        input = input_data.transpose([2, 0, 1])
        input = input[np.newaxis, :, :, :]
        input_handle.reshape([1, input.shape[0], input.shape[1], input.shape[2]])  # 这里不对输入tensor作任何处理
        input_handle.copy_from_cpu(input)
        # 运行predictor
        start = time.time()
        self.retrieval_predictor.run()
        print("run for "+str(time.time()-start))
        # 获取输出
        output_names = self.retrieval_predictor.get_output_names()
        output_handle = self.retrieval_predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()  # numpy.ndarray类型
        output = output_data.squeeze().astype("uint8")
        print(output_data.shape)
        print(output.shape)
        # print(np.bincount(output).shape)
        stats = zip(self.config.retrieval_category, np.bincount(output.reshape(-1))[:len(self.config.retrieval_category)])
        output_img = self._get_pseudo_color_map(output)
        output_img.save("output.png")
        return output_img,stats



class MapImageHelper:
    @staticmethod
    def getImage(x, y, size=[4,4]):
        if len(size) != 2:
            raise Exception("map image size should have exactly two coordinate")
        im_list = []
        for i in range(0,size[0]):
            for j in range(0,size[1]):
                url = Map_url_template.format(x+i,y+j)
                response = requests.get(url)
                data = response.content
                image = Image.open(BytesIO(data)).convert('RGB')
                im = np.array(image)
                im_list.append(im)
        im_list = np.array(im_list).reshape([4, 4, 256, 256, 3])
        im_list = np.transpose(im_list, [1, 2, 0, 3, 4])
        im_list = im_list.reshape([1024, 1024, 3])
        im = Image.fromarray(im_list)
        return im

