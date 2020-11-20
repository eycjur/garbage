from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from .forms import UploadPictureForm
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

def index(request):
    params = {
        "form":UploadPictureForm()
    }
    return render(request, "garbage/index.html", params)


def result(request, num=0):
    if num:
        img = BASE_DIR + "/static/garbage/media/images/" + ["temp1.jpg", "temp2.jpg"][num-1]

    else:
        form = UploadPictureForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data["img"]
        else:
            params = {
                "form":UploadPictureForm()
            }
            return render(request, "garbage/index.html", params)

    pred = predict(img)

    params = {
        "img":img,
        "pred":pred
    }
    return render(request, "garbage/result.html", params)

def predict(img):
    # 読み込み
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.preprocessing import image
    from keras.models import model_from_json
    from PIL import Image

    model = model_from_json(open(BASE_DIR + "/static/model.json").read())
    model.load_weights(BASE_DIR + '/static/param.hdf5')

    img_width, img_height = 150, 150
    img = Image.open(img)
    img.save(BASE_DIR + "/static/garbage/media/images/image.png")
    img = np.array(img.resize((img_width, img_height)))
    classes = ['不燃ごみ', '包装容器プラスチック類', '可燃ごみ', '有害ごみ', '資源品']
    days = ["第2・4木曜日", "水曜日", "火・金曜日", "第1・3金曜日", "第1・3金曜日"]

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # 画像の人物を予測
    pred = model.predict(x)[0]
    # 結果を表示する
    np.set_printoptions(suppress=True)
    pred_list = [[c, "{:.2f}".format(s), d] for (c, s, d) in zip(classes, pred*100, days)]
    pred_list = sorted(pred_list, key=lambda x:x[1], reverse=True)
    return pred_list
