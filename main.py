from recognize import Recognize as R

trainner_yml='trainner/trainner_4.yml'
cascade_path = 'haarcascade_frontalface_default.xml'
names = ['aidai', 'anhu ', 'axin', 'baibaihe', 'baijingting', 'baike', 'baobeier', 'baojianfeng', 'Anbei', '', 'Luoxiang', 'Johnson', 'Macron', 'Yangmi', 'Wangsicong']

a = R(trainner_yml, cascade_path, names)
# 识别文件夹中的照片
# a.img_recognize("D:/pycharm/facetest/baojianfeng")
# 摄像头识别
# a.cam_recongnize()
# 口罩识别
a.mask_recognize()
# 显示结果
a.show_result()
