# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2022//04//03
"""

import paddle.inference as paddle_infer
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
        self.sort_config = paddle_infer.Config(config.sort_model_path, config.sort_param_path)
        self.sort_predictor = paddle_infer.create_predictor(self.sort_config)

        self.retrieval_config = paddle_infer.Config(config.retrieval_model_path, config.retrieval_param_path)
        if config.enable_gpu:
            self.retrieval_config.enable_use_gpu(memory_pool_init_size_mb=2048,device_id=0)
        self.retrieval_predictor = paddle_infer.create_predictor(self.retrieval_config)
        #
        # self.detection_config = paddle_infer.Config(config.detection_model_path, config.detection_param_path)
        # self.detection_predictor = paddle_infer.create_predictor(self.detection_config)

    @classmethod
    def _generate_translucent_background(cls,binary_mask, color_mask):
        assert binary_mask.shape == color_mask.shape
        result_array = np.zeros([binary_mask.shape[0], binary_mask.shape[1], 4])
        result_array[:,:,:3] = color_mask
        result_array[:,:,3] = binary_mask*255-128

        return Image.fromarray(np.uint8(result_array),"RGBA")

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
    def _get_pseudo_color_map(cls, pred, color_map=None, translucent_background=True):
        pred_mask = PIL.Image.fromarray(pred.astype("uint8"), mode='P')
        if color_map is None:
            color_map = Predictor._get_color_map_list(num_classes=256)
        pred_mask.putpalette(color_map)
        pred_mask = pred_mask.convert('RGBA')
        pred_mask = np.array(pred_mask)
        if translucent_background:
            pred_mask[np.where(pred == 0)] = (0, 0, 0, 0)
        return PIL.Image.fromarray(pred_mask)

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

    def contrast_predict(self, np_img,target_img):
        input_names = self.contrast_predictor.get_input_names()
        input_handle1 = self.contrast_predictor.get_input_handle(input_names[0])
        input_handle2 = self.contrast_predictor.get_input_handle(input_names[1])

        img1 = np.array(np_img.resize((1024, 1024)))
        img2 = np.array(target_img.resize((1024, 1024)))

        img1 = self._normalize(img1)[np.newaxis, :, :, :]
        img2 = self._normalize(img2)[np.newaxis, :, :, :]

        input_handle1.reshape([1, 3, 1024, 1024])
        input_handle1.copy_from_cpu(img1)
        input_handle2.reshape([1, 3, 1024, 1024])
        input_handle2.copy_from_cpu(img2)
        # 运行predictor
        begin_time = time.time()
        self.contrast_predictor.run()
        end_time = time.time()
        print("predict time: %f" % (end_time - begin_time))
        # 获取输出
        output_names = self.contrast_predictor.get_output_names()
        output_handle = self.contrast_predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()  # numpy.ndarray类型
        output_data = output_data.reshape(1024, 1024)
        im = Image.fromarray(output_data.astype('uint8') * 255)
        return im, np.bincount(im.reshape(-1))


    def sort_predict(self, file):
        input_names = self.sort_predictor.get_input_names()
        input_handle = self.sort_predictor.get_input_handle(input_names[0])
        input_data = np.array(file).astype("float32")
        input_shape = input_data.shape
        input_data = self._normalize(input_data)
        input = input_data.transpose([2, 0, 1])
        input = input[np.newaxis, :, :, :]
        input_handle.reshape([1, input.shape[0], input.shape[1], input.shape[2]])  # 这里不对输入tensor作任何处理
        input_handle.copy_from_cpu(input)
        # 运行predictor
        self.sort_predictor.run()
        # 获取输出
        output_names = self.sort_predictor.get_output_names()
        output_handle = self.sort_predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()  # numpy.ndarray类型
        output = output_data.squeeze().astype("uint8")
        output_img = self._get_pseudo_color_map(output,translucent_background=False)
        if output_img.size != file.size:
            output_img = output_img.resize((file.size[0], file.size[1]), Image.NEAREST)
        return output_img, np.bincount(output.reshape(-1))[:len(self.config.sort_category)]

    def detection_predict(self, file):
        pass

    def retrieval_predict(self, file):
        input_names = self.retrieval_predictor.get_input_names()
        input_handle = self.retrieval_predictor.get_input_handle(input_names[0])
        # input_data = np.array(file.resize([1024,1024])).astype("float32")
        input_data = np.array(file).astype("float32")
        input_shape = input_data.shape
        input_data = self._normalize(input_data)
        input = input_data.transpose([2, 0, 1])
        input = input[np.newaxis, :, :, :]
        input_handle.reshape([1, input.shape[0], input.shape[1], input.shape[2]])  # 这里不对输入tensor作任何处理
        input_handle.copy_from_cpu(input)
        # 运行predictor
        self.retrieval_predictor.run()
        # 获取输出
        output_names = self.retrieval_predictor.get_output_names()
        output_handle = self.retrieval_predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()  # numpy.ndarray类型
        output = output_data.squeeze().astype("uint8")
        output_img = self._get_pseudo_color_map(output,translucent_background=True)
        if output_img.size != file.size:
            output_img = output_img.resize((file.size[0], file.size[1]), Image.NEAREST)
        return output_img, np.bincount(output.reshape(-1))[1]


def toRad(value):
    return value * np.pi / 180


class MapImageHelper:
    @staticmethod
    def getImage(x1, y1, x2, y2, zoom=18):
        # x1, y1, x2, y2, start_x, start_y, end_x, end_y = MapImageHelper.coordinate_transfer(x1, y1, x2, y2, zoom)
        x1, y1, x2, y2, start_x, start_y, end_x, end_y = MapImageHelper.coordinate_transfer(x1, y2, x2, y1, zoom)
        # im_list = []
        # for i in range(x1, x2 + 1):
        #     for j in range(y1, y2 + 1):
        #         url = Map_url_template.format(i, j)
        #         response = requests.get(url)
        #         data = response.content
        #         image = Image.open(BytesIO(data)).convert('RGB')
        #         im = np.array(image)
        #         print(im.shape)
        #         im_list.append(im)
        # im_list = np.array(im_list).reshape([x2 - x1 + 1, y2 - y1 + 1, 256, 256, 3])
        # im_list = np.transpose(im_list, [1, 2, 0, 3, 4])
        # im_list = im_list.reshape([256 * (x2 - x1 + 1), 256 * (y2 - y1 + 1), 3])

        im_list = np.zeros([256 * (y2 - y1 + 1), 256 * (x2 - x1 + 1), 3])
        for i in range(x1, x2 + 1):
            for j in range(y1, y2 + 1):
                url = Map_url_template.format(i, j)
                response = requests.get(url)
                data = response.content
                image = Image.open(BytesIO(data)).convert('RGB')
                im = np.array(image)
                # im_list.append(im)
                im_list[(j - y1) * 256:(j - y1 + 1) * 256, (i - x1) * 256:(i - x1 + 1) * 256, :] = im

        start_idx = np.floor(im_list.shape[0] * (start_x - x1) / (x2 - x1 + 1)).astype(np.int32)
        end_idx = np.floor(im_list.shape[0] * (end_x - x1) / (x2 - x1 + 1)).astype(np.int32)

        start_idy = np.floor(im_list.shape[1] * (start_y - y1) / (y2 - y1 + 1)).astype(np.int32)
        end_idy = np.floor(im_list.shape[1] * (end_y - y1) / (y2 - y1 + 1)).astype(np.int32)
        im_list = im_list[start_idy: end_idy, start_idx: end_idx]

        im = Image.fromarray(im_list)
        return im

    @staticmethod
    def coordinate_transfer(x1, y1, x2, y2, zoom=18):
        start_x = (x1 + 180) / 360 * (1 << zoom)
        xtile_1 = np.floor((x1 + 180) / 360 * (1 << zoom)).astype(np.int32).item(0)
        start_y = (1 - np.log(np.tan(toRad(y1)) + 1 / np.cos(toRad(y1))) / np.pi) / 2 * (1 << zoom)
        ytile_1 = np.floor((1 - np.log(np.tan(toRad(y1)) + 1 / np.cos(toRad(y1))) / np.pi) / 2 * (1 << zoom)).astype(
            np.int32).item(0)

        end_x = (x2 + 180) / 360 * (1 << zoom)
        xtile_2 = np.floor((x2 + 180) / 360 * (1 << zoom)).astype(np.int32).item(0)
        end_y = (1 - np.log(np.tan(toRad(y2)) + 1 / np.cos(toRad(y2))) / np.pi) / 2 * (1 << zoom)
        ytile_2 = np.floor((1 - np.log(np.tan(toRad(y2)) + 1 / np.cos(toRad(y2))) / np.pi) / 2 * (1 << zoom)).astype(
            np.int32).item(0)
        return xtile_1, ytile_1, xtile_2, ytile_2, start_x, start_y, end_x, end_y

