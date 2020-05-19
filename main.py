from recognize import Recognize as R

trainner_yml='trainner/zhuoqun_trainner.yml'
cascade_path = 'haarcascade_frontalface_default.xml'
names = ['axin', 'zhuhao', 'Trump', 'Obama', 'songbinbin', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun', 'xuzhuoqun']

a=R(trainner_yml,cascade_path,names)
a.cam_recongnize()
a.show_result()