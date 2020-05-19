from recognize import Recognize as R

trainner_yml='trainner/trainner.yml'
cascade_path = 'haarcascade_frontalface_default.xml'
names = [' ', ' ', 'Trump', 'Obama', ' ', ' ', ' ', 'Liucixin', 'Anbei', '', 'Luoxiang', 'Johnson', 'Macron', 'Yangmi', 'Wangsicong']

a=R(trainner_yml,cascade_path,names)
a.img_recognize("C:/now/face/face-recognize/img")
a.show_result()
