import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import cloudpickle


# 分類するクラス
classes = glob("../train/*")
classes = [c.split("\\", 1)[-1] for c in classes]
nb_classes = len(classes)

#画像の大きさを設定
img_width, img_height = 224, 224

# 画像フォルダの指定
train_dir = '../train'
val_dir = '../val'

#バッチサイズ
batch_size = 16


# 水増し処理
train_data_transform = transforms.Compose([
    transforms.Resize((img_width, img_height), interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_data_transform = transforms.Compose([
    transforms.Resize((img_width, img_height), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=train_data_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

val_data = torchvision.datasets.ImageFolder(root=val_dir, transform=val_data_transform)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

# VGG16
model = models.vgg16(pretrained=True)
# model.classifier = nn.Sequential(
#     nn.Linear(in_features=512 * 7 * 7, out_features=256),
#     nn.ReLU(inplace=True),
#     nn.Dropout(),
#     nn.Linear(in_features=256, out_features=128),
#     nn.ReLU(inplace=True),
#     nn.Dropout(),
#     nn.Linear(in_features=128, out_features=nb_classes),
# )
model.classifier[6] = nn.Linear(in_features=4096, out_features=nb_classes)
model.train()
criterion = nn.CrossEntropyLoss()

params_update = []
params_update_name = "classifier.6"
for name, params in model.named_parameters():
    if params_update_name in name:
        params.required_grad = True
        params_update.append(params)
    else:
        params.required_grad = False

optimizer = optim.SGD([
    {"params":params_update, "lr":1e-4}
], momentum = 0.9)

# 50エポック学習
for epoch in range(3):
    running_loss = 0.0
    correct_num = 0
    total_num = 0
    for i, (inputs, labels) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        correct_num_temp = (predicted==labels).sum()
        correct_num += correct_num_temp.item()
        total_num += batch_size
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 経過の出力
    print(f'epoch:{epoch + 1:d} loss: {running_loss / 100:.3f} acc: {correct_num * 100 / total_num:.3f}')


# 保存
with open('../static/model.pkl', 'wb') as f:
    cloudpickle.dump(model, f)

with open('../static/model.pkl', 'rb') as g:
    model2 = cloudpickle.load(g)

with open('https://drive.google.com/file/d/1Io-Nu4clhZD_55GmEp26Nj_pSlLD0b6w/download', 'rb') as g:
    model2 = cloudpickle.load(g)






















# 読み込み
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import model_from_json
model = model_from_json(open("model.json").read())
model.load_weights('param.hdf5')

img_width, img_height = 150, 150
classes = ['不燃ごみ', '包装容器プラスチック類', '可燃ごみ', '有害ごみ', '資源品']

filename = "val/不燃ごみ/IMG_20201108_114503.jpg"
img = image.load_img(filename, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要
# これを忘れると結果がおかしくなるので注意
x = x / 255.0
# 表示
plt.imshow(img.resize((120,160)))
plt.show()
plt.close()

# 画像の人物を予測
pred = model.predict(x)[0]
# 結果を表示する
result = {c:s for (c, s) in zip(classes, pred*100)}
result = sorted(result.items(), key=lambda x:x[1], reverse=True)
print(result)
result
