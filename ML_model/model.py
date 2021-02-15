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

# 画像の大きさを設定
img_width, img_height = 160, 160

# 画像フォルダの指定
train_dir = '../train'
val_dir = '../val'

# バッチサイズ
batch_size = 16


# 水増し方法の指定
train_gen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1. / 255,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

val_gen = ImageDataGenerator(rescale=1. / 255)

# ジェネレーターを生成
train_generator = train_gen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)

val_generator = val_gen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)


# datasetを生成
train_data = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_types=("float32", "float32")
).repeat()
val_data = tf.data.Dataset.from_generator(
        lambda: val_generator,
        output_types=("float32", "float32")
)


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
          optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
          metrics=['acc'])
vgg_model.summary()

# FIXME:エラー出るので最初だけ学習
vgg_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=1,
    validation_data=val_generator,
    validation_steps=len(val_generator))

history = vgg_model.fit(
    train_data,
    steps_per_epoch=len(train_generator) * 5,
    epochs=30,
    validation_data=val_data,
    validation_steps=len(val_generator))


# acc, val_accのプロット
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

