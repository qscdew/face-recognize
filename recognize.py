import os

import cv2
import numpy
from PIL import Image
class Recognize:

    def __init__(self,trainner_yml,cascade_path,names):
        self.trainner_yml=trainner_yml #训练好的模型路径
        self.cascade_path=cascade_path #人脸分类器的路径地址
        self.names=names #与id号码对应的用户名 格式为数组

        #准备好识别方法
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        #使用训练模型
        self.recognizer.read(trainner_yml)

        #调用人脸分类器
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

        # 加载一个字体，用于识别后，在图片上标注出对象的名字
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        #保存结果
        self.result=[]

    def show_result(self):
        #print(self.result)
        for i in self.result:
            print(i)

    #调用摄像头识别
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
                if confidence < 120:
                    name = self.names[idnum]
                    confidence = "{0}%".format(round(120 - confidence))
                else:
                    name = "unknown"
                    confidence = "{0}%".format(round(120 - confidence))

                # 输出检验结果以及用户名
                cv2.putText(img, str(name), (x + 5, y - 5), self.font, 1, (0, 0, 255), 1)
                cv2.putText(img, str(confidence), (x + 5, y + h), self.font, 1, (0, 255, 0), 1)

                # 识别到结果 释放资源
                cam.release()
                cv2.destroyAllWindows()

                res = {"image": "cam_image",
                       "name": str(name),
                       "id": idnum,
                       "confidence": confidence}

                self.result.append(res)

    # 识别图片
    def img_recognize(self, path):


        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        # 遍历图片路径，导入图片和id添加到list中
        for image_path in image_paths:
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

                # 计算出一个检验结果
                if confidence < 120:
                    name = self.names[idnum]
                    confidence = "{0}%".format(round(120 - confidence))
                else:
                    name = "unknown"
                    confidence = "{0}%".format(round(120 - confidence))

                # 输出检验结果以及用户名
                cv2.putText(img, str(name), (x + 5, y - 5), self.font, 1, (0, 0, 255), 1)
                cv2.putText(img, str(confidence), (x + 5, y + h), self.font, 1, (0, 255, 0), 1)

                cv2.destroyAllWindows()

                res={"image":image_path,
                     "name":str(name),
                     "id":idnum,
                     "confidence":confidence}

                self.result.append(res)


