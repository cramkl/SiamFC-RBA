from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms

import xlrd
import xlwt
import math

__all__ = ['TrackerSiamFC']

class ContourStructrue:
    def __init__(self):
        self.num = 0        # 数量
        self.length = []    # 每个轮廓的长度
        self.area = []      # 每个轮廓的面积
        self.max_value = [] # 每个轮廓峰值的值
        self.pos_x = []     # 每个轮廓内峰值的x坐标
        self.pos_y = []     # 每个轮廓内峰值的y坐标

class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

        self.frame_cnt = 0
        self.init_APK = 0
        self.workbook = xlwt.Workbook()
        self.worksheet = self.workbook.add_sheet('peak')
        self.worksheet2 = self.workbook.add_sheet('peak1')
        self.worksheet_train = self.workbook.add_sheet('train')
        self.worksheet_contour = self.workbook.add_sheet('contour')
        self.worksheet_distanceValue = self.workbook.add_sheet('value')

        self.response_sizeRatio_list = []
        self.response_dist_list = []
        self.response_merge_flag = 0
        self.response_alarm_flag = 0

        self.delta_value = []
        self.delta_value_cnt = 0
        self.contour_list = []
        self.contour_cnt = 0
        self.MaxCoutourNum = 9999
        self.search_size = 255
        self.warning_state = []
        self.warning_cnt = 0

        self.trace_point_x = []
        self.trace_point_y = []
        self.trace_point_cnt = 0
        self.trace_pointMax = 40
        #在测试模式下，不会保存和显示新增加的测试图像和数据
        self.test_mode = 0


    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127, #127
            'instance_sz': 255, #255
            'context': 0.5,
            # inference parameters
            'scale_num': 3,  #3
            'scale_step': 1.0175,  #1.0375
            'scale_lr': 0.59,
            'scale_penalty': 0.9745, #0.9745
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 0, #32
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    def multi_peak_analyze(self, responses,id):

        response = responses[id]
        response -= response.min()
        response /= response.sum() + 1e-16
        max = response.max()
        min = response.min()
        #print('response', id, '-max=', response.max(), '-min=', response.min(), '-sum=', response.sum())
        sum = 0
        sum2 = 0
        res = response.astype(np.float32)
        for m in range(res.shape[0]):
            for n in range(res.shape[1]):
                sum = sum + (res[m][n] - min) * (res[m][n] - min)
                if res[m][n] < 1 / res.shape[0] / res.shape[1]:
                    sum2 = sum2 + res[m][n]
        #self.worksheet.write(self.frame_cnt, 2, sum2)
        mean = sum / res.shape[0] / res.shape[1]
        mean2 = sum2/res.shape[0]
        sum = 0
        for m in range(res.shape[0]):
            for n in range(res.shape[1]):
                sum = sum + (res[m][n] - mean2) * (res[m][n] - mean2)
        mean = sum / res.shape[0] / res.shape[1]
        APK = ((max - min) * (max - min)) / mean
        self.worksheet.write(self.frame_cnt, 0, APK)
        self.worksheet.write(self.frame_cnt, 1, (float)(max))

        if self.frame_cnt == 0:
            self.init_APK = APK
            print('#############Init APK = ', APK, '##############')
        else:
            if APK < 3*self.init_APK/4:
                print('#############Warning!!! APK = ', APK, '##############')
        return None

    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        # 数据集中的目标框的格式为“左上角点+长和宽”，转换成“矩形框中心+长和宽”格式
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]
        self.second_center, self.second_target_sz = box[:2], box[2:]

        # create hanning window
        # 创建hanning窗，np.hanning窗为一行N列向量，为余弦或者高斯分布
        # np.outer和np.hanning配合使用，即使用两个一维的hanning窗，生成一个二维的高斯分布平面
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        # 对应原文中的目标尺度变化，原文为：1.025^(-2,-1,0,1,2)
        # 这里程序里默认的是 1.0375^(-1,0,1)
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        # context 对应论文公式（7）中的2p,  p = (w+h)/4，这里默认self.cfg.context = 0.5 = 2p
        context = self.cfg.context * np.sum(self.target_sz)
        # np.prod(self.target_sz + context) = (w+2p)*(h+2p)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        # 如果待跟踪目标为(w*h)的举行，那么先得到一个边长是sqrt((w+2p)*(h+2p))的正方形
        # 再将此正方形resize成一个边长是exemplar_sz（127）的正方形
        # 此时得到的就是127*127的模板图像
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        #test to save exemplar image
        #img_temp = z.copy()
        img_temp = cv2.cvtColor(z, cv2.COLOR_RGB2BGR)
        cv2.imwrite("exemplar_image.jpg",img_temp)
        
        # exemplar features
        # 创建张量
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)
    
    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,  #modified 20220118  out_size=self.cfg.instance_sz  self.search_size
            border_value=self.avg_color) for f in self.scale_factors]
        if self.frame_cnt == 382:
            img_temp = cv2.cvtColor(x[0], cv2.COLOR_RGB2BGR)
            cv2.imwrite("search_image.jpg", img_temp)

        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        
        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        # 上采样成272*272尺寸的response图
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16

        #self.multi_peak_analyze(responses,scale_id)

        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz

        #self.center += disp_in_image  修改
        ############################################################
        #########新增加代码
        analyze_flag = 0
        analyze_test_flag = 0 #测试标记，如果想测试原始的SiamFC算法，又想纪录数据时，标志位为1，否则为0
        if analyze_flag == 1:
            if self.test_mode == 0:
                if self.frame_cnt == 381:
                    self.response_datasave(response)
            disp = disp_in_image.ravel()
            # 计算偏移距离，即前后两次的目标中心偏移是否过大
            offset_dist = (disp[0] ** 2 + disp[1] ** 2) ** (1 / 2)
            if self.test_mode == 0:
                self.worksheet.write(self.frame_cnt, 2, float(offset_dist))
            # response分析跟踪是否丢失
            track_state,_ = self.response_analyse(response, loc, img)
            # if self.response_alarm_flag == 0 or offset_dist < 35:
            if analyze_test_flag == 0:
                if track_state != 2 and self.response_alarm_flag == 0:# and offset_dist < 35:
                    self.center += disp_in_image
            else:
                self.center += disp_in_image
        else:
            self.center += disp_in_image
        #########新增加代码结束
        ###########################################################

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        #self.frame_cnt = self.frame_cnt + 1
        return box
    
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box

        boxes2 = np.zeros((frame_num, 4))
        boxes2[0] = box

        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :], frame_num= self.frame_cnt-1)

        self.workbook.save("peak.xls")
        return boxes, times
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
            self.worksheet_train.write(epoch + 1, 1, loss)
        print('======Train Done=====')
        self.workbook.save("peak.xls")
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels
    def response_datasave(self, response):

        response_save = response.copy()
        data = response_save.ravel()
        if self.frame_cnt == 2:
            for i in range(272):
                for j in range(272):
                    temp = data[i*272+j]
                    i = 271-i
                    if j+3 < 256:
                        self.worksheet.write(i, j+3, temp)
                    else:
                        self.worksheet2.write(i, j + 3 - 256, temp)
            print(data)
        return None
    def response_analyse(self, response, loc, origin_img):

        track_state = 1
        self.response_merge_flag = 0
        response_new = response.copy()
        response_new = 255 * (response_new - response_new.min()) / (response_new.max() - response_new.min())
        img = response_new.astype(np.uint8)
        img_gray = img.copy()

        _, gray = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY) #125

        if self.test_mode == 0:
            file_name = './results/response_nodraw/frame_r_{}.jpg'.format(self.frame_cnt)
            cv2.imwrite(file_name, img)
            file_name = './results/response_binary/frame_b_{}.jpg'.format(self.frame_cnt)
            cv2.imwrite(file_name, gray)

        y = loc[0]
        x = loc[1]

        distance_value = ((4*255)**2+x**2+y**2)**(1/2)

        first_max = 0
        first_max_i = 0
        first_max_j = 0
        second_max = 0
        second_max_i = 0
        second_max_j = 0

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        _,contours,hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #for c in contours:
        cv2.drawContours(img,contours,-1,(0,0,255),2)

        if len(contours) > 1:
            #冒泡法排序，面积大的在前，小的在后
            for m in range(1,len(contours)):
                for n in range(0,len(contours)-m):
                    if contours[n].size < contours[n+1].size:
                        contours[n],contours[n+1] = contours[n+1],contours[n]
        #删除小轮廓
        #for item in contours:
        #    if item.size < 30:
        #        contours.remove(item)

        if len(contours) > 0:
            if self.contour_cnt < self.MaxCoutourNum:
                contour_temp = ContourStructrue()
                contour_temp.num = len(contours)
                for i in range(len(contours)):
                    contour_temp.area.append(cv2.contourArea(contours[i]))
                    contour_temp.length.append(cv2.arcLength(contours[i],True))
                    x0, y0, w0, h0 = cv2.boundingRect(contours[i])
                    temp_max = 0
                    temp_max_i = 0
                    temp_max_j = 0
                    for i0 in range(x0, x0 + w0, 1):
                        for j0 in range(y0, y0 + h0, 1):
                            if img_gray[j0][i0] > temp_max:
                                temp_max = img_gray[j0][i0]
                                temp_max_i = i0
                                temp_max_j = j0
                    contour_temp.max_value.append(temp_max)
                    contour_temp.pos_x.append(temp_max_i)
                    contour_temp.pos_y.append(temp_max_j)
                self.contour_list.append(contour_temp)
                self.contour_cnt = self.contour_cnt + 1

        self.response_alarm_flag = 0
        alarm_flag = 0
        merge_flag = 0
        min_dist = 10000
        second_max_i = 0
        second_max_j = 0

        #for test breakpoint
        if self.frame_cnt==720:
            test1_flag = 1
        if len(contours) > 1:
            #冒泡法排序，面积大的在前，小的在后
            for m in range(1,len(contours)):
                for n in range(0,len(contours)-m):
                    if contours[n].size < contours[n+1].size:
                        contours[n],contours[n+1] = contours[n+1],contours[n]


            """
            ContA = contours[0].ravel()
            ContB = contours[1].ravel()
            conA_min_x = 0
            conA_min_y = 0
            conB_min_x = 0
            conB_min_y = 0
            """

            """
            # 第一大轮廓的外接矩形
            x0, y0, w0, h0 = cv2.boundingRect(contours[0])
            cv2.rectangle(img, (x0, y0), (x0 + w0, y0 + h0), (255, 0, 0), 1)
            first_max = 0
            first_max_i = 0
            first_max_j = 0
            for i0 in range(x0, x0 + w0, 1):
                for j0 in range(y0, y0 + h0, 1):
                    if img_gray[j0][i0] > first_max:
                        first_max = img_gray[j0][i0]
                        first_max_i = i0
                        first_max_j = j0

            #第二大轮廓的外接矩形
            x1,y1,w,h = cv2.boundingRect(contours[1])
            cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),1)
            second_max = 0
            second_max_i = 0
            second_max_j = 0
            for ii in range(x1,x1+w,1):
                for jj in range(y1,y1+h,1):
                    if img_gray[jj][ii] > second_max:
                        second_max = img_gray[jj][ii]
                        second_max_i = ii
                        second_max_j = jj
            """

            #计算每个轮廓内峰值及位置
            contour_max_list = []
            contour_max_i_list=[]
            contour_max_j_list=[]
            for i in range(len(contours)):
                x1, y1, w, h = cv2.boundingRect(contours[i])
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 1)
                tmp_max = 0
                tmp_max_i = 0
                tmp_max_j = 0
                for ii in range(x1, x1 + w, 1):
                    for jj in range(y1, y1 + h, 1):
                        if img_gray[jj][ii] > tmp_max:
                            tmp_max = img_gray[jj][ii]
                            tmp_max_i = ii
                            tmp_max_j = jj
                contour_max_list.append(tmp_max)
                contour_max_i_list.append(tmp_max_i)
                contour_max_j_list.append(tmp_max_j)

            # 冒泡法排序，峰值大的在前，小的在后
            for m in range(1, len(contour_max_list)):
                for n in range(0, len(contour_max_list) - m):
                    if contour_max_list[n] < contour_max_list[n + 1]:
                        contours[n], contours[n + 1] = contours[n + 1], contours[n]
                        contour_max_list[n], contour_max_list[n + 1] = contour_max_list[n + 1], contour_max_list[n]
                        contour_max_i_list[n], contour_max_i_list[n + 1] = contour_max_i_list[n + 1], contour_max_i_list[n]
                        contour_max_j_list[n], contour_max_j_list[n + 1] = contour_max_j_list[n + 1], contour_max_j_list[n]

            first_max = contour_max_list[0]
            second_max = contour_max_list[1]
            first_max_i = contour_max_i_list[0]
            first_max_j = contour_max_j_list[0]
            second_max_i = contour_max_i_list[1]
            second_max_j = contour_max_j_list[1]

            # 第一大轮廓的外接矩形
            x0, y0, w0, h0 = cv2.boundingRect(contours[0])
            cv2.rectangle(img, (x0, y0), (x0 + w0, y0 + h0), (255, 0, 0), 1)
            # 第二大轮廓的外接矩形
            x1, y1, w, h = cv2.boundingRect(contours[1])
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 1)
            #两个轮廓的面积之和
            total_area = w0*h0+w*h

            ContA = contours[0].ravel()
            ContB = contours[1].ravel()
            conA_min_x = 0
            conA_min_y = 0
            conB_min_x = 0
            conB_min_y = 0


            # 遍历轮廓上的每一个点，计算两个轮廓最近的点
            for i in range(int(len(ContA)/2)):
                conA_x = int(ContA[2*i])
                conA_y = int(ContA[2*i+1])
                for j in range(int(len(ContB)/2)):
                    conB_x = int(ContB[2 * j])
                    conB_y = int(ContB[2 * j + 1])
                    dist = ((conA_x-conB_x)**2+(conA_y-conB_y)**2)**(1/2)
                    if dist < min_dist:
                        min_dist = dist
                        conA_min_x = conA_x
                        conA_min_y = conA_y
                        conB_min_x = conB_x
                        conB_min_y = conB_y
            size_ratio = contours[0].size / contours[1].size
            if size_ratio < 3 and min_dist < 50:
                alarm_flag = 1
                self.response_alarm_flag = 1
            cv2.line(img, (conA_min_x, conA_min_y), (conB_min_x, conB_min_y), (0, 255, 0), 2)
            cv2.line(img, (second_max_i, second_max_j), (first_max_i, first_max_j), (0, 0, 255), 2)
            # 画中心位置十字
            cv2.line(img, (first_max_i - 10, first_max_j), (first_max_i + 10, first_max_j), (0, 0, 255), 2)
            cv2.line(img, (first_max_i, first_max_j - 10), (first_max_i, first_max_j + 10), (0, 0, 255), 2)
            cv2.line(img, (second_max_i - 10, second_max_j), (second_max_i + 10, second_max_j), (0, 255, 0), 2)
            cv2.line(img, (second_max_i, second_max_j - 10), (second_max_i, second_max_j + 10), (0, 255, 0), 2)

            distance_value = ((4*(float(first_max)-float(second_max)))**2+(float(first_max_i)-float(second_max_i))**2+(float(first_max_j)-float(second_max_j))**2)**(1/2)

            if self.delta_value_cnt < 5:
                self.delta_value.append(float(first_max)-float(second_max))
                self.delta_value_cnt = self.delta_value_cnt + 1
            else:
                for i in range(0,4):
                    self.delta_value[i] = self.delta_value[i+1]
                self.delta_value[4] = float(first_max)-float(second_max)
            #print('=======Frame:',self.frame_cnt,'=',first_max-second_max,'==========')
            if self.frame_cnt > 5:
                state = self.trend_analyse(self.delta_value)
                if self.test_mode == 0:
                    print('merging state:',state)
                if state == 1:
                    self.response_merge_flag = 1
        else:
            # 画中心位置十字
            cv2.line(img, (x - 10, y), (x + 10, y), (0, 0, 255), 2)
            cv2.line(img, (x, y - 10), (x, y + 10), (0, 0, 255), 2)

        if len(self.response_dist_list) < 5:
            self.response_dist_list.append(min_dist)
        else:
            self.response_dist_list[0] = self.response_dist_list[1]
            self.response_dist_list[1] = self.response_dist_list[2]
            self.response_dist_list[2] = self.response_dist_list[3]
            self.response_dist_list[3] = self.response_dist_list[4]
            self.response_dist_list[4] = min_dist
            if self.response_dist_list[0] > self.response_dist_list[4] and \
               self.response_dist_list[1] > self.response_dist_list[4] and \
               self.response_dist_list[2] > self.response_dist_list[4] and \
               self.response_dist_list[3] > self.response_dist_list[4] and \
               self.response_dist_list[1] < self.response_dist_list[0] and \
               self.response_dist_list[2] < self.response_dist_list[0] and \
               self.response_dist_list[3] < self.response_dist_list[0]:
                    merge_flag = 1
                    #self.response_merge_flag = 1

        if self.test_mode == 0:
            self.worksheet_distanceValue.write(self.frame_cnt, 0, distance_value)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'frame{}'.format(self.frame_cnt)
        img = cv2.putText(img, text, (20, 40), font, 0.8, (255, 255, 255), 1)
        if alarm_flag == 1:
            alart_test = 'ALARM'
            img = cv2.putText(img, alart_test, (180, 40), font, 0.8, (0, 0, 255), 1)
        if self.response_merge_flag == 1:
            merge_test = 'MERGING'
            img = cv2.putText(img, merge_test, (180, 100), font, 0.8, (0, 255, 0), 1)

        if self.test_mode == 0:
            cv2.imshow("response", img)
            cv2.imshow("binary", gray)
            file_name = './results/response/frame_{}.jpg'.format(self.frame_cnt)
            cv2.imwrite(file_name,img)
            self.worksheet.write(self.frame_cnt, 0, float(x))
            self.worksheet.write(self.frame_cnt, 1, float(y))
            if self.contour_list[self.contour_cnt-1].num > 0:
                num_temp = self.contour_list[self.contour_cnt-1].num
                self.worksheet_contour.write(self.contour_cnt-1, 0, self.contour_list[self.contour_cnt-1].num)
                for i in range(num_temp):
                    self.worksheet_contour.write(self.contour_cnt-1, 1+5*i, int(self.contour_list[self.contour_cnt-1].length[i]))
                    self.worksheet_contour.write(self.contour_cnt-1, 2+5*i, int(self.contour_list[self.contour_cnt-1].area[i]))
                    self.worksheet_contour.write(self.contour_cnt-1, 3+5*i, int(self.contour_list[self.contour_cnt-1].max_value[i]))
                    self.worksheet_contour.write(self.contour_cnt-1, 4+5*i, int(self.contour_list[self.contour_cnt-1].pos_x[i]))
                    self.worksheet_contour.write(self.contour_cnt-1, 5+5*i, int(self.contour_list[self.contour_cnt-1].pos_y[i]))

        self.frame_cnt = self.frame_cnt + 1

        second_loc = loc
        origin_img = cv2.cvtColor(origin_img,cv2.COLOR_RGB2BGR)
        temp_warning_flag = 0
        if len(contours) > 1:
            second_loc = list(second_loc)
            second_loc[0] = second_max_j
            second_loc[1] = second_max_i
            second_loc = tuple(second_loc)
            self.draw_response_peak(origin_img, second_loc,2)
            #分析两个峰值之间的距离和角度关系
            cen_dist1 = ((first_max_i-self.cfg.instance_sz/2)**2 + (first_max_j-self.cfg.instance_sz/2)**2)**(1/2)
            cen_dist2 = ((second_max_i - self.cfg.instance_sz/2) ** 2 + (second_max_j - self.cfg.instance_sz/2) ** 2)**(1/2)

            if cen_dist1 > cen_dist2:
                temp1 = first_max
                first_max = second_max
                second_max = temp1
                temp1 = first_max_i
                first_max_i = second_max_i
                second_max_i = temp1
                temp1 = first_max_j
                first_max_j = second_max_j
                second_max_j = temp1

            pt1_x = first_max_i
            pt1_y = first_max_j
            pt1_z = first_max
            pt2_x = second_max_i
            pt2_y = second_max_j
            pt2_z = second_max

            #for test
            if self.test_mode == 0:
                self.worksheet2.write(self.frame_cnt, 0, int(first_max_i))
                self.worksheet2.write(self.frame_cnt, 1, int(first_max_j))
                self.worksheet2.write(self.frame_cnt, 2, int(first_max))
                self.worksheet2.write(self.frame_cnt, 3, int(second_max_i))
                self.worksheet2.write(self.frame_cnt, 4, int(second_max_j))
                self.worksheet2.write(self.frame_cnt, 5, int(second_max))

            if self.test_mode == 0:
                print('Frame:', self.frame_cnt, 'max-second=', float(pt1_z) - float(pt2_z))
                print('area_ratio=', total_area / self.cfg.instance_sz / self.cfg.instance_sz)

            if cen_dist1 < cen_dist2 and first_max > second_max:
                dist = ((first_max_i-second_max_i)**2 + (first_max_j-second_max_j)**2)**(1/2)
                theta = math.atan((first_max-second_max)/(dist))
                self.search_size = 255

                if self.test_mode == 0:
                    self.worksheet_distanceValue.write(self.contour_cnt - 1, 2,float(theta))
                    self.worksheet_distanceValue.write(self.contour_cnt - 1, 3, 255)
            elif cen_dist1 < 2*self.cfg.instance_sz/5 or cen_dist2 < 2*self.cfg.instance_sz/5:

                dist = ((pt1_x - pt2_x) ** 2 + (pt1_y - pt2_y) ** 2) ** (1 / 2)
                tan_data = 0
                if dist > 0.00001:
                    tan_data = (float(pt1_z) - float(pt2_z)) / float(dist)
                theta = math.atan(tan_data)

                new_sz = int(255*(1+3*math.sin(math.fabs(theta))))  #3
                if self.test_mode == 0:
                    print('theta=',theta)
                    print('dist=',dist)
                if (float(pt1_z) - float(pt2_z)) < 0 and theta >-0.06:#(float(pt1_z) - float(pt2_z)) > -10: # -30
                    self.search_size = new_sz
                    temp_warning_flag = 1
                elif theta <=-0.06:#float(pt1_z) - float(pt2_z) <= -5:
                    temp_warning_flag = 2
                if self.test_mode == 0:
                    self.worksheet_distanceValue.write(self.contour_cnt - 1, 2, float(theta))
                    self.worksheet_distanceValue.write(self.contour_cnt - 1, 3, float(new_sz))

                #if total_area > 0.20*self.cfg.instance_sz*self.cfg.instance_sz:
                #    temp_warning_flag = 1
                #    print('######################!!!Total area error!*****************************')
                #if theta < 0:

                #    print('new_sz = ', new_sz)

            #print('Two theta=',theta)
        else:
            #print('self.cfg:', type(self.cfg))
            #self.cfg.update({'instance_sz': 255})
            if self.test_mode == 0:
                self.worksheet_distanceValue.write(self.contour_cnt - 1, 2, 0)
                self.worksheet_distanceValue.write(self.contour_cnt - 1, 3, 255)
            self.search_size = 255
            # 分析两个峰值之间的距离和角度关系
            dist = ((first_max_i - second_max_i) ** 2 + (first_max_j - second_max_j) ** 2) ** (1 / 2)
            if dist != 0:
                theta = math.atan((first_max - second_max) / (dist))
                #print('One theta=', theta)

        if self.warning_cnt < 5:
            self.warning_state.append(temp_warning_flag)
            self.warning_cnt = self.warning_cnt + 1
        else:
            self.warning_state[0] = self.warning_state[1]
            self.warning_state[1] = self.warning_state[2]
            self.warning_state[2] = self.warning_state[3]
            self.warning_state[3] = self.warning_state[4]
            self.warning_state[4] = temp_warning_flag
            if self.warning_state[4] == 1:
                #self.warning_state[2] == 1 and self.warning_state[3] == 1 and \
                track_state = 2
                if self.test_mode == 0:
                    print('****Frame:',self.frame_cnt,'******tracking state 2******')
            if self.warning_state[4] == 2:
                track_state = 3
                if self.test_mode == 0:
                    print('****Frame:', self.frame_cnt, '******tracking state 3******')

        trace_x,trace_y = self.draw_response_peak(origin_img, loc,1)
        if self.trace_point_cnt < self.trace_pointMax:
            self.trace_point_x.append(trace_x)
            self.trace_point_y.append(trace_y)
            self.trace_point_cnt = self.trace_point_cnt + 1
        else:
            for i in range(len(self.trace_point_x)-1):
                self.trace_point_x[i] = self.trace_point_x[i + 1]
                self.trace_point_y[i] = self.trace_point_y[i + 1]
            self.trace_point_x[len(self.trace_point_x) - 1] = trace_x
            self.trace_point_y[len(self.trace_point_y) - 1] = trace_y
        for i in range(self.trace_point_cnt):
            cv2.circle(origin_img,(self.trace_point_x[i],self.trace_point_y[i]),3,(255,0,0),2)

        if self.test_mode == 0:
            cv2.imshow("new",origin_img)

        return track_state,second_loc

    #数据趋势分析，判断一段数据是呈现上升趋势还是下降趋势
    def trend_analyse(self, data):

        state = 0
        data_len = 0
        data_len = len(data)
        dist = []
        if data_len > 0:
            # 计算由N（=15）个点组成的数据的起始点和终点组成的直线方程
            data_temp1 = float(data[data_len-1])
            data_temp2 = float(data[0])
            k = (data_temp1-data_temp2)/data_len
            b = data_temp1 - k*data_len
            # 计算除起始点和终点外其他所有点到这条直线之间的距离
            for i in range(1,data_len):
                dist_temp = abs(k*(i+1)-data[i]+b)/(k*k+1)**(1/2)
                dist.append(dist_temp)
        sum_data = 0
        for i in range(len(dist)):
            sum_data = sum_data + dist[i]
        #计算距离的均值，均值越小，说明和直线的拟合度越好
        mean_val = 0
        if len(dist)> 0:
            mean_val = sum_data / len(dist)
        #print('k=', k, ',b=', b, 'mean_val=',mean_val)
        # 如果斜率k小于一定值并且距离均值也比较小，那说明整个数据呈比较明显的下降趋势
        if k < -6 and mean_val < 2:
            state = 1
        # 如果斜率k大于一定值并且距离均值也比较小，那说明整个数据呈比较明显的上升趋势
        elif k > 6 and mean_val < 2:
            state = 2
        '''if self.frame_cnt == 380 or self.frame_cnt == 381 or self.frame_cnt == 382:
            print('##################')
            print('k=',k,',b=',b)
            print(dist)
            print('##################')'''
        return state

    # 轮廓合并和分裂分析，判断几个轮廓间是产生了合并和是分裂
    def merge_split_analyze(self, data):

        state = 0

        return state

    def draw_response_peak(self, img, loc, order):

        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
                           self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
                        self.scale_factors[0] / self.cfg.instance_sz
        temp_center = self.center + disp_in_image

        if order == 1:
            cv2.line(img, (int(temp_center[1] - 10), int(temp_center[0])), (int(temp_center[1] + 10), int(temp_center[0])), (0, 0, 255), 2)
            cv2.line(img, (int(temp_center[1]), int(temp_center[0] - 10)), (int(temp_center[1]), int(temp_center[0] + 10)), (0, 0, 255), 2)
        else:
            cv2.line(img, (int(temp_center[1] - 10), int(temp_center[0])),(int(temp_center[1] + 10), int(temp_center[0])), (0, 255, 0), 2)
            cv2.line(img, (int(temp_center[1]), int(temp_center[0] - 10)),(int(temp_center[1]), int(temp_center[0] + 10)), (0, 255, 0), 2)

        x = int(temp_center[1])
        y = int(temp_center[0])
        return x,y

