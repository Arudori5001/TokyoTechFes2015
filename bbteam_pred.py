#coding: utf-8

import os, sys
import cv2
import PIL.Image as Image
from chainer import FunctionSet
import chainer.functions as F
import glob
import numpy as np
import pickle

import Learning


def preprocess(input_file):
    """
    @summary: 予測する画像の整形を行う。
    画像はデスクトップから移動して取得する
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
    os.system(com)

    #make LibSVM inputs
    #libsvm_file = "input.libsvm"
    #f = open(libsvm_file, 'w')
    img = Image.open(resized_file)
    img.convert("RGB")

    return img, face_file


def display_result(result_class, result_prob, plobs, face_file): 
    """
    @summary: 予測結果を表示する
    @param result_class: int    予測されたラベル(0-Origin)
    result_prob: float    予測されたラベルの確信度(事後確率)
    plobs: np.array<float>    各ラベルの確信度リスト
    face_file: PIL.Image    予測結果を表示するための画像
    """
    
    dataset_dir = "dataset"
    cluster_names = [os.path.relpath(d, dataset_dir) for d in glob.glob(os.path.join(dataset_dir, "*"))]
    
    print("predicted class : " + cluster_names[result_class])
    print("certainty factor of predicted class : {0}".format(res_prob))
    print("certainty factors of each classes : ")
    for c,p in zip(cluster_names, plobs):
        print(c + " :")
        print("\t{0:>5.2f}%".format(p*100))
    
    
    msg = "You belong to %s! (%i%%)"%(cluster_names[result_class] , int(100*result_prob))
    msg_file = "msg.jpg"
    com = """convert -font Helvetica-Bold -pointsize 40 -gravity north -annotate 0x0-50+50 "%s" -fill white %s %s"""%(
        msg, face_file, msg_file)
    os.system(com)

    com = "display %s"%(msg_file)
    os.system(com)


def create_toy_model(n_output):
    """
    @summary: トイモデルをつくる
    """
    model = FunctionSet(conv1=F.Convolution2D(3,32,5,pad=2),
        conv2=F.Convolution2D(32,32,5,pad=2),
        conv3=F.Convolution2D(32,64,5,pad=2),
        fl5=F.Linear(960, 64),
        fl6=F.Linear(64, n_output))
    
    return model
    
    
def create_toy_image(shape):
    """
    @summary: トイデータをつくる
    """
    
    n_pixels = np.prod(shape)
    imgs = np.arange(n_pixels).reshape(shape[0], shape[1], shape[2], shape[3])
    imgs = np.array(imgs, dtype=np.float32)
    
    return imgs


def predict(img):
    """
    @summary: 画像から予測を行う
    @param img: PIL.Image 予測を行う画像
    @return: int    予測されたラベル(0-Origin)
        float    予測されたラベルの確信度(事後確率)
        np.array<float>    　各ラベルの確信度リスト
    """

    n_channel = 3
    shape = (1,n_channel,36,46)
    
    model = None
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
    
    record = Learning.img_to_record(img, 0)
    inp = record.input
    inp = np.array([inp], dtype=np.float32)
            
    oup = Learning.forward(model, inp)
    sm = F.softmax(oup)
    probs = sm.data[0]
    res = np.argmax(probs)
    res_prob = np.max(probs)
    
    return res, res_prob, probs


if __name__ == "__main__":
    input_file = sys.argv[1]
    img, face_file = preprocess(input_file)
    
    res, res_prob, probs = predict(img)
    
    display_result(res, res_prob, probs, face_file)
    