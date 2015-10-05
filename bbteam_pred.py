#coding: utf-8

import os, sys
import cv2
from chainer import FunctionSet
import chainer.functions as F
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import PIL.Image as Image
import pylab

import Learning


def preprocess(input_file):
    """
    @summary: 予測する画像の整形を行う
    @param input_file: str 予測する画像のパス
    @return: PIL.Image 整形後の画像
    PIL.Image 顔認識の枠の書かれた画像
    """
    
    
    #com = "mv ~/Desktop/*jpg in.jpg"
    #os.system(com)
    #input_file = "in.jpg"
    
    #顔認識をする
    cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
    white = (255, 255, 255) 
    red   = (0, 0, 255) 
    image = cv2.imread(input_file)
    image_gray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(100, 100))

    if len(facerect) > 0:
        rect = facerect[0]
        cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), white, thickness=2)

        xcenter = rect[0] + (rect[2]/2)
        ycenter = rect[1] + (rect[3]/2)
        
        ssizex = int((rect[2])*1.3) #1.3 is magic number!!!
        sposx = max( (xcenter-(ssizex/2)), 0 )
        ssizey = int(ssizex*(46.0/36.0))
        sposy = max( (ycenter-(ssizey/2)), 0 )

        cv2.rectangle(image, (sposx, sposy), (sposx+ssizex, sposy+ssizey), red, thickness=1)
                   
        face_file = "detected.jpg"
        cv2.imwrite(face_file, image)
        
        com = "display %s"%(face_file)
        os.system(com)

    else:
        print "cannot detect!"
        sys.exit()


    #crop face
    cropped_file = "crop.jpg"
    com = "convert -crop %dx%d+%d+%d %s %s"%(ssizex, ssizey, sposx, sposy, input_file, cropped_file)
    os.system(com)

    #resize
    resized_file = "res.jpg"
    com = "convert -resize 36x46 %s %s"%(cropped_file, resized_file)
    os.system(com)

    com = "display %s"%(resized_file)
    #os.system(com)

    img = Image.open(resized_file)
    img.convert("RGB")

    return img, face_file


def predict(img):
    """
    @summary: 画像から予測を行う
    @param img: PIL.image   予測を行う画像
    @return: np.array<float>    　各ラベルの確信度(事後確率)のリスト
    """

    n_channel = 3
    shape = (1,n_channel,36,46)
    
    model = None
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
    
    record = Learning.img_to_record(img, 0)
    inp = np.array([record.input], dtype=np.float32)
            
    oup = Learning.forward(model, inp)
    sm = F.softmax(oup)
    probs = sm.data[0]
    
    return probs


def show_chart(table):
    """
    @summary: 棒グラフを表示する
    @param table: list<tuple<str, float>> ラベルと数値のリスト
    """

    labels = [x[0] for x in table]
    values = [x[1] for x in table]
    width = 0.5
    margin = 0.8
    barcolor = "#3D57CC"
    axis_bgcolor = "#F6F6F6"

    plt.rcParams.update({
        "font.size":    20,
        "grid.color":   barcolor,
        "grid.linestyle":   "-"})
    pylab.figure(facecolor="w")
    pylab.axes(axisbg=axis_bgcolor)
    plt.barh(range(len(table)), values, width, align="center", alpha=1, color=barcolor, edgecolor=barcolor)
    plt.yticks(np.array(range(len(table))) , labels)
    pylab.xlabel('Probabilities')
    pylab.ylabel('Categoriy Names')
    plt.axis([0, 1, -1, len(table)])
    plt.grid(True)
    pylab.subplots_adjust(left=0.25, bottom=0.15)
    plt.show()


def display_result(plobs, face_file): 
    """
    @summary: 予測結果を表示する
    @param plobs: np.array<float>    各ラベルの確信度リスト
    face_file: PIL.Image    予測結果を表示するための画像
    """
    
    dataset_dir = "dataset"
    category_names = [os.path.relpath(d, dataset_dir) for d in glob.glob(os.path.join(dataset_dir, "*"))]
    result_category = np.argmax(probs)
    result_prob = np.max(probs)
    
    print("predicted categoriy : " + category_names[result_category])
    print("certainty factor of predicted categoriy : {0}".format(result_prob))
    print("certainty factors of each categories : ")
    for c,p in zip(category_names, plobs):
        print(c + " :")
        print("\t{0:>5.2f}%".format(p*100))
    
    
    msg = "You belong to %s! (%i%%)"%(category_names[result_category] , int(100*result_prob))
    msg_file = "msg.jpg"
    com = """convert -font Helvetica-Bold -pointsize 40 -gravity north -annotate 0x0-50+50 "%s" -fill white %s %s"""%(
        msg, face_file, msg_file)
    os.system(com)

    com = "display %s &"%(msg_file)
    os.system(com)

    table = sorted(zip(category_names, probs), key=lambda x:x[1])
    show_chart(table)


if __name__ == "__main__":
    input_file = sys.argv[1]
    img, face_file = preprocess(input_file)
    
    probs = predict(img)
    
    display_result(probs, face_file)
    