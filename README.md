# TokyoTechFes2015

## Requirements
* ImageMagick  
macならば http://cactuslab.com/imagemagick/ など
* OpenCV  
`pip install cv2`
* pillow  
`pip install pillow`
* Chainer  
`pip install chainer`

## Installation
```
git clone https://github.com/ritsu73/TokyoTechFes2015.git
cd TokyoTechFes2015
unzip dataset.zip
```

## Usage
### 予測
```
python bbteam_pred.py （予測したい画像のパス）
```

### 学習
1. `dataset`フォルダ内に判別したいクラスの名前のフォルダをそれぞれ作成する
2. 作成したフォルダにそれぞれのクラスの画像を入れる
3. ```python Learning.py （学習データ数） (テストデータ数)```

ちなみに`dataset.zip`を解凍したものを使う場合、学習データ数=520、テストデータ数=130 くらいが妥当.