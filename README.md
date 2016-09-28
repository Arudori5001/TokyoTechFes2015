# TokyoTechFes2015
工大祭向けの画像判別デモスクリプト

## Requirements
* ImageMagick  
macならば http://cactuslab.com/imagemagick/ など
* OpenCV  
`pip install cv2`
* pillow  
`pip install pillow`
* Chainer(ver 1.3.2)  
`pip install chainer==1.3.2`
* matplotlib  
`pip install matplotlib`

## Installation
```
git clone https://github.com/Arudori5001/TokyoTechFes2015.git
cd TokyoTechFes2015
unzip dataset.zip
pip install ...
```


## Usage
### 予測
```
python bbteam_pred.py （予測したい画像のパス）
```

### 学習
1. `dataset`フォルダ内に判別したいクラスの名前のフォルダをそれぞれ作成する
2. 作成したフォルダにそれぞれのクラスの画像を入れる
3. ```python Learning.py （学習に使うデータ数） (テストに使うデータ数)```

ちなみに`dataset.zip`を解凍したものは全部で680個の画像が入っているので`python Learning.py 520 130`くらいが良いかもしれない.