from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from .forms import UploadPictureForm
from PIL import Image


def index(request):
    params = {
        "form":UploadPictureForm()
    }
    return render(request, "garbage/index.html", params)


def result(request):
    form = UploadPictureForm(request.POST, request.FILES)
    if form.is_valid():
        img = form.cleaned_data["img"]
        pred = predict(img)
    else:
        params = {
            "form":UploadPictureForm()
        }
        return render(request, "garbage/index.html", params)

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

    model = model_from_json(open("../model.json").read())
    model.load_weights('../param.hdf5')

    img_width, img_height = 150, 150
    img = Image.open(img)
    img.save("image.png")
    img = np.array(img.resize((img_width, img_height)))
    classes = ['不燃ごみ', '包装容器プラスチック類', '可燃ごみ', '有害ごみ', '資源品']

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # 画像の人物を予測
    pred = model.predict(x)[0]
    # 結果を表示する
    pred_dict = {c:s for (c, s) in zip(classes, pred*100)}
    pred_dict = sorted(pred_dict.items(), key=lambda x:x[1], reverse=True)
    return pred_dict
