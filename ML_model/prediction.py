import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import model_from_json


model = model_from_json(open("model.json").read())
model.load_weights('param.hdf5')

img_width, img_height = 160 ,160
classes = ['不燃ごみ', '包装容器プラスチック類', '可燃ごみ', '有害ごみ', '資源品']

filename = "../val/不燃ごみ/IMG_20201108_114503.jpg"
img = image.load_img(filename, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# 学習時に正規化したので同じ処理が必要
x = x / 255.0
plt.imshow(img.resize((120, 160)))
plt.show()
# plt.close()

# 画像の人物を予測
pred = model.predict(x)[0]
# 結果を表示する
result = {c:s for (c, s) in zip(classes, pred*100)}
result = sorted(result.items(), key=lambda x:x[1], reverse=True)
print(result)
