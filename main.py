from recognize import Recognize as R

trainner_yml='trainner/trainner_3.yml'
cascade_path = 'haarcascade_frontalface_default.xml'
names = ['axin', 'zhuhao', 'Trump', 'Obama', 'songbinbin', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun']

a=R(trainner_yml,cascade_path,names)
a.img_recognize("C:/now/face/face-recognize/img")
a.show_result()
