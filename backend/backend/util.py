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
from io import BytesIO
import cv2
from PIL import Image, ImageDraw, ImageFont
Map_url_template = "https://webst01.is.autonavi.com/appmaptile?style=6&x={}&y={}&z=18&scl=1"


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top - 22), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class WindowGenerator:
    def __init__(self, h, w, ch, cw, si=1, sj=1):
        self.h = h
        self.w = w
        self.ch = ch
        self.cw = cw
        if self.h < self.ch or self.w < self.cw:
            raise NotImplementedError
        self.si = si
        self.sj = sj
        self._i, self._j = 0, 0
    def __next__(self):
        # 列优先移动（C-order）
        if self._i > self.h:
            raise StopIteration
        bottom = min(self._i+self.ch, self.h)
        right = min(self._j+self.cw, self.w)
        top = max(0, bottom-self.ch)
        left = max(0, right-self.cw)

        if self._j >= self.w-self.cw:
            if self._i >= self.h-self.ch:
                # 设置一个非法值，使得迭代可以early stop
                self._i = self.h+1
            self._goto_next_row()
        else:
            self._j += self.sj
            if self._j > self.w:
                self._goto_next_row()
        return slice(top, bottom, 1), slice(left, right, 1)
    def __iter__(self):
        return self
    def _goto_next_row(self):
        self._i += self.si
        self._j = 0
        
def recons_prob_map(patches, ori_size, window_size, stride):
    """从裁块结果重建原始尺寸影像"""
    h, w = ori_size
    win_gen = WindowGenerator(h, w, window_size, window_size, stride, stride)
    prob_map = np.zeros((h,w), dtype=np.float)
    cnt = np.zeros((h,w), dtype=np.float)
    # XXX: 需要保证win_gen与patches具有相同长度。此处未做检查
    for (rows, cols), patch in zip(win_gen, patches):
        prob_map[rows, cols] += patch
        cnt[rows, cols] += 1
    prob_map /= cnt
    return prob_map

class Predictor:
    def __init__(self, config):
        self.config = config
        self.contrast_config = paddle_infer.Config(config.constrast_model_path, config.constrast_param_path)
        self.contrast_predictor = paddle_infer.create_predictor(self.contrast_config)
        self.sort_config = paddle_infer.Config(config.sort_model_path, config.sort_param_path)
        self.sort_predictor = paddle_infer.create_predictor(self.sort_config)

        self.retrieval_config = paddle_infer.Config(config.retrieval_model_path, config.retrieval_param_path)
        if config.enable_gpu:
            self.retrieval_config.enable_use_gpu(memory_pool_init_size_mb=2048,device_id=0)
        self.retrieval_predictor = paddle_infer.create_predictor(self.retrieval_config)

        self.detection_config = paddle_infer.Config(config.detection_model_path, config.detection_param_path)
        self.detection_predictor = paddle_infer.create_predictor(self.detection_config)

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

    def contrast_predict(self, old_img, new_img):
        input_names = self.contrast_predictor.get_input_names()
        input_handle1 = self.contrast_predictor.get_input_handle(input_names[0])
        input_handle2 = self.contrast_predictor.get_input_handle(input_names[1])
        predictor = self.contrast_predictor
        img1 = np.array(old_img.resize((1024, 1024)))
        img2 = np.array(new_img.resize((1024, 1024)))
        ori_size = (1024, 1024)
        img1 = self._normalize(img1)[np.newaxis, :, :, :].transpose((0,3,1,2))
        img2 = self._normalize(img2)[np.newaxis, :, :, :].transpose((0,3,1,2))
        STRIDE = 256
        W = 384
        patch_pairs = []
        for rows, cols in WindowGenerator(*ori_size, W, W, STRIDE, STRIDE):
            patch_pairs.append((img1[:, :,rows, cols], img2[:, :, rows, cols]))
        input_handle1.reshape([1, 3, W, W])
        input_handle2.reshape([1, 3, W, W])
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        result_list = []
        predict_time = 0
        for im1, im2 in patch_pairs:
            input_handle1.copy_from_cpu(im1)
            input_handle2.copy_from_cpu(im2)
            begin_time = time.time()
            predictor.run()
            end_time = time.time()
            predict_time += end_time - begin_time
            print("predict time: %f" % (end_time - begin_time))
            output_data = output_handle.copy_to_cpu() # numpy.ndarray类型
            output_data = output_data.reshape((W, W))
            result_list.append(output_data)
        prob_map = recons_prob_map(result_list, ori_size, W, STRIDE)
    # 对概率图进行阈值分割，得到二值变化图
        output = (prob_map>=0.5).astype('int32')
        output_img = self._get_pseudo_color_map(output,translucent_background=True)
        return output_img, np.bincount(output.reshape(-1)), predict_time


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
        begin_time = time.time()
        self.sort_predictor.run()
        end_time = time.time()
        predict_time = end_time - begin_time
        # 获取输出
        output_names = self.sort_predictor.get_output_names()
        output_handle = self.sort_predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()  # numpy.ndarray类型
        output = output_data.squeeze().astype("uint8")

        output_img = self._get_pseudo_color_map(output,color_map=self.config.sort_color_map,translucent_background=False)
        if output_img.size != file.size:
            output_img = output_img.resize((file.size[0], file.size[1]), Image.NEAREST)
        return output_img, np.bincount(output.reshape(-1))[:len(self.config.sort_category)], predict_time

    def detection_predict(self, file):
        input_names = self.detection_predictor.get_input_names()
        shape_handler = self.detection_predictor.get_input_handle(input_names[0])
        image_handler = self.detection_predictor.get_input_handle(input_names[1])
        scale_handler = self.detection_predictor.get_input_handle(input_names[2])

        org_size = file.size
        img = file.resize((640, 640))
        scale_factor = np.array([640.0 / org_size[0], 640.0 / org_size[1]]).astype('float32')
        img = np.array(img)
        shape = img.shape
        img = self._normalize(img)[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype('float32')
        shape_handler.copy_from_cpu(np.array([[shape[0], shape[1]]]).astype('float32'))
        image_handler.reshape((1, 3, shape[0], shape[1]))
        image_handler.copy_from_cpu(img)
        scale_handler.copy_from_cpu(scale_factor)
        begin_time = time.time()
        self.detection_predictor.run()
        end_time = time.time()
        print("predict time: %f" % (end_time - begin_time))
        # 获取输出
        output_names = self.detection_predictor.get_output_names()
        output_handle = self.detection_predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()  # numpy.ndarray类型

        mask = np.ones_like(np.array(file))*255
        statistic = [0 for i in range(0, 15)]
        for item in output_data:
            c, p, l, t, r, b = item
            color_tuple = tuple(self.config.detection_color_list[3*int(c):3*int(c)+3])
            if p > 0.5:
                cv2.rectangle(mask, (int(l), int(t)), (int(r), int(b)), color_tuple, 2)
                mask = cv2ImgAddText(mask, self.config.detection_label_list[int(c)], int(l), int(t), color_tuple)
                statistic[int(c)] += 1

        transparent_result = np.zeros((org_size[0], org_size[1], 4))
        transparent_result[:,:,:3] = mask
        transparent_mask = np.sum(mask,2)
        transparent_mask[transparent_mask != 765] = 255
        transparent_mask[transparent_mask == 765] = 0

        transparent_result[:,:,3] = transparent_mask
        result = Image.fromarray(np.uint8(transparent_result),"RGBA")
        return result, statistic, end_time - begin_time

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
        begin_time = time.time()
        self.retrieval_predictor.run()
        end_time = time.time()
        predict_time = end_time - begin_time
        # 获取输出
        output_names = self.retrieval_predictor.get_output_names()
        output_handle = self.retrieval_predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()  # numpy.ndarray类型
        output = output_data.squeeze().astype("uint8")
        output_img = self._get_pseudo_color_map(output,translucent_background=True)
        if output_img.size != file.size:
            output_img = output_img.resize((file.size[0], file.size[1]), Image.NEAREST)
        return output_img, np.bincount(output.reshape(-1)), predict_time


def toRad(value):
    return value * np.pi / 180


class MapImageHelper:
    @staticmethod
    def getImage(x1, y1, x2, y2, zoom=18):
        x1, y1, x2, y2, start_x, start_y, end_x, end_y = MapImageHelper.coordinate_transfer(x1, y1, x2, y2, zoom)
        im_list = np.zeros([256 * (y2 - y1 + 1), 256 * (x2 - x1 + 1), 3])
        for i in range(x1, x2 + 1):
            for j in range(y1, y2 + 1):
                url = Map_url_template.format(i, j)
                response = requests.get(url)
                data = response.content
                image = Image.open(BytesIO(data)).convert('RGB')
                im = np.array(image)
                im_list[(j - y1) * 256:(j - y1 + 1) * 256, (i - x1) * 256:(i - x1 + 1) * 256, :] = im

        start_idx = np.floor(im_list.shape[1] * (start_x - x1) / (x2 - x1 + 1)).astype(np.int32)
        end_idx = np.floor(im_list.shape[1] * (end_x - x1) / (x2 - x1 + 1)).astype(np.int32)

        start_idy = np.floor(im_list.shape[0] * (start_y - y1) / (y2 - y1 + 1)).astype(np.int32)
        end_idy = np.floor(im_list.shape[0] * (end_y - y1) / (y2 - y1 + 1)).astype(np.int32)
        im_list = im_list[start_idy: end_idy, start_idx: end_idx]

        im = Image.fromarray(np.uint8(im_list),"RGB")
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

