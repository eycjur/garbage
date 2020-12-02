from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf


# 分類するクラス
classes = glob("../train/*")
classes = [c.split("\\", 1)[-1] for c in classes]
nb_classes = len(classes)

#画像の大きさを設定
img_width, img_height = 160, 160

# 画像フォルダの指定
train_dir = '../train'
val_dir = '../val'

#バッチサイズ
batch_size = 16


# 水増し処理
train_datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1. / 255,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

# ジェネレーターを生成
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)


# ジェネレーターを生成
train_data = tf.data.Dataset.from_generator(lambda:train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True), output_types=("float32", "float32")).repeat()
val_data = tf.data.Dataset.from_generator(lambda:val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True), output_types=("float32", "float32"))


# VGG16
input_tensor = Input(shape=(img_width, img_height, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# 全結合層
top_model = Sequential()
top_model.add(BatchNormalization(input_shape=vgg16.output_shape[1:]))
top_model.add(Flatten())
top_model.add(Dropout(0.5))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))
top_model.summary()

vgg_model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# 重みを固定
for layer in vgg_model.layers[:19]:
    layer.trainable = False

vgg_model.compile(loss='categorical_crossentropy',
          optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
          metrics=['acc'])
vgg_model.summary()

# エラー出るので最初だけ学習
history = vgg_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=1,
    validation_data=val_generator,
    validation_steps=len(val_generator))

history = vgg_model.fit(
    train_data,
    steps_per_epoch=len(train_generator) * 5,
    epochs=50,
    validation_data=val_data,
    validation_steps=len(val_generator))


#acc, val_accのプロット
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
plt.ylabel("acc")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.savefig("acc")
plt.close()

plt.plot(history.history["loss"], label="loss", ls="-", marker="o")
plt.plot(history.history["val_loss"], label="val_loss", ls="-", marker="x")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.savefig("loss")
plt.close()

# 保存
open("model.json", 'w').write(vgg_model.to_json())
vgg_model.save_weights('param.hdf5')


























# 読み込み
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
# 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要
# これを忘れると結果がおかしくなるので注意
x = x / 255.0
# 表示
plt.imshow(img.resize((120,160)))
plt.show()
# plt.close()

# 画像の人物を予測
pred = model.predict(x)[0]
# 結果を表示する
result = {c:s for (c, s) in zip(classes, pred*100)}
result = sorted(result.items(), key=lambda x:x[1], reverse=True)
print(result)
result
