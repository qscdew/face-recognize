import cv2
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

    def show_result(self):
        print(self.result_name,self.result_confidence)

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
                if confidence < 80:
                    idum = self.names[idnum]
                    confidence = "{0}%".format(round(100 - confidence))
                else:
                    idum = "unknown"
                    confidence = "{0}%".format(round(100 - confidence))

                # 输出检验结果以及用户名
                cv2.putText(img, str(idum), (x + 5, y - 5), self.font, 1, (0, 0, 255), 1)
                cv2.putText(img, str(confidence), (x + 5, y + h), self.font, 1, (0, 255, 0), 1)

                # 识别到结果 释放资源
                cam.release()
                cv2.destroyAllWindows()

                self.result_name=str(idum)
                self.result_confidence=str(confidence)
                return 1



    # 识别图片
    def img_recognize(self,img_path):
        idum=""
        confidence=0

        #to do

        self.result_name=str(idum)
        self.result_confidence=str(confidence)
        return 1