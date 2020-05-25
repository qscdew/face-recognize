# face-recognize
基于python和opencv库实现window、Linux和树莓派下的人脸识别。


2020/05/24更新  main函数新增口罩识别。

## 环境
```
Package               Version
--------------------- --------
Pillow                7.1.2
PyYAML                5.3.1
boltons               20.1.0
future	              0.18.2	
numpy                 1.18.4
opencv-contrib-python 3.3.0.10
opencv-python         4.2.0.34

pip                   19.0.3
setuptools            40.8.0


torch	              1.5.0	
torchvision	      0.6.0	
utils	              1.0.1	
```

## 运行
```
python main.py
```
可通过函数调用三个不同的功能
# 识别文件夹中的照片
# a.img_recognize("D:/pycharm/facetest/baojianfeng")
# 摄像头识别
# a.cam_recongnize()
# 口罩识别
a.mask_recognize()


