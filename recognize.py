import os

import cv2
import numpy as np
from PIL import Image
import argparse
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.pytorch_loader import load_pytorch_model, pytorch_inference



class Recognize:

    def __init__(self, trainner_yml, cascade_path, names):
        self.trainner_yml = trainner_yml
        # 训练好的模型路径
        self.cascade_path = cascade_path
        # 人脸分类器的路径地址
        self.names = names
        # 与id号码对应的用户名 格式为数组

        # 准备好识别方法
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        # 使用训练模型
        self.recognizer.read(self.trainner_yml)

        # 调用人脸分类器
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

        # 加载一个字体，用于识别后，在图片上标注出对象的名字
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # 保存结果
        self.result = []

        # 口罩检测模型加载
        self.model_path = 'models/model360.pth'
        self.feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
        self.anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        self.anchor_ratios = [[1, 0.62, 0.42]] * 5

    def show_result(self):
        # print(self.result)
        for i in self.result:
            print(i)

    # 调用摄像头识别
    def cam_recongnize(self):
        cam = cv2.VideoCapture(0)
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:

            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 识别人脸
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH))
            )
            # 进行校验
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                idnum, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
                # idnum和confidence表示predict返回的标签和置信度，confidence越小匹配度越高，0表示完全匹配

                # 计算出一个检验结果
                if confidence < 80:
                    name = self.names[idnum]
                    confidence = "{0}%".format(round(100 - confidence))
                else:
                    name = "unknown"
                    confidence = "{0}%".format(round(100 - confidence))

                # 输出检验结果以及用户名
                cv2.putText(img, str(name), (x + 5, y - 5), self.font, 1, (0, 0, 255), 1)
                cv2.putText(img, str(confidence), (x + 5, y + h), self.font, 1, (0, 255, 0), 1)

                # cv2.putText(img, str(idnum), (x + 5, y + h + h / 2), self.font, 1, (255, 255, 255), 1)

            # 展示结果
            cv2.imshow('camera', img)
            k = cv2.waitKey(20)
            if k == 27:
                break
        # 识别到结果 释放资源
        cam.release()
        cv2.destroyAllWindows()



    # 识别图片
    def img_recognize(self, path):


        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        # 遍历图片路径，导入图片和id添加到list中
        for image_path in image_paths:
            count = 0
            # 通过图片路径将其转换为灰度图片
            img = cv2.imread(image_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            minW = 0.1 * img.shape[1]
            minH = 0.1 * img.shape[0]
            faces = self.face_cascade.detectMultiScale(
                gray_img,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH))
            )

            for (x, y, w, h) in faces:

                idnum, confidence = self.recognizer.predict(gray_img[y:y + h, x:x + w])
                # idnum和confidence表示predict返回的标签和置信度，confidence越小匹配度越高，0表示完全匹配
                count += 1
                # 计算出一个检验结果
                if confidence < 100:
                    name = self.names[idnum]
                    confidence = "{0}%".format(round(100 - confidence))
                else:
                    name = "unknown"
                    confidence = "{0}%".format(round(100 - confidence))

                # 输出检验结果以及用户名
                # cv2.putText(img, str(name), (x + 5, y - 5), self.font, 1, (0, 0, 255), 1)
                # cv2.putText(img, str(confidence), (x + 5, y + h), self.font, 1, (0, 255, 0), 1)

                # cv2.destroyAllWindows()

                res={"image": image_path,
                     "name": str(name),
                     "id": idnum,
                     "confidence": confidence}

                self.result.append(res)
            if count == 0:
                res = {"image:"+image_path + "     Not find face !"}
                self.result.append(res)

        # 口罩识别
    def mask_recognize(self):
        model = load_pytorch_model(self.model_path)
        # generate anchors
        anchors = generate_anchors(self.feature_map_sizes, self.anchor_sizes, self.anchor_ratios)
        anchors_exp = np.expand_dims(anchors, axis=0)

        id2class = {0: 'Mask', 1: 'NoMask'}

        def inference(image,
                      conf_thresh=0.5,
                      iou_thresh=0.4,
                      target_shape=(160, 160),
                      draw_result=True,
                      show_result=True
                      ):
            output_info = []
            height, width, _ = image.shape
            image_resized = cv2.resize(image, target_shape)
            image_np = image_resized / 255.0  # 归一化到0~1
            image_exp = np.expand_dims(image_np, axis=0)

            image_transposed = image_exp.transpose((0, 3, 1, 2))

            y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
            # remove the batch dimension, for batch is always 1 for inference.
            y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
            y_cls = y_cls_output[0]
            # To speed up, do single class NMS, not multiple classes NMS.
            bbox_max_scores = np.max(y_cls, axis=1)
            bbox_max_score_classes = np.argmax(y_cls, axis=1)

            # keep_idx is the alive bounding box after nms.
            keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                         bbox_max_scores,
                                                         conf_thresh=conf_thresh,
                                                         iou_thresh=iou_thresh,
                                                         )

            for idx in keep_idxs:
                conf = float(bbox_max_scores[idx])
                class_id = bbox_max_score_classes[idx]
                bbox = y_bboxes[idx]
                # clip the coordinate, avoid the value exceed the image boundary.
                xmin = max(0, int(bbox[0] * width))
                ymin = max(0, int(bbox[1] * height))
                xmax = min(int(bbox[2] * width), width)
                ymax = min(int(bbox[3] * height), height)

                if draw_result:
                    if class_id == 0:
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
                output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

            if show_result:
                Image.fromarray(image).show()
            return output_info

        cap = cv2.VideoCapture(0)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        if not cap.isOpened():
            raise ValueError("Video open failed.")
            return
        status = True
        while status:
            status, img_raw = cap.read()
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            if (status):
                inference(img_raw,
                          conf_thresh=0.5,
                          iou_thresh=0.5,
                          target_shape=(360, 360),
                          draw_result=True,
                          show_result=False)
            cv2.imshow('image', img_raw[:, :, ::-1])
            k = cv2.waitKey(20)
            if k == 27:
                break
        cv2.destroyAllWindows()
