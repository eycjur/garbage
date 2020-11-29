from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from django.views.decorators.csrf import requires_csrf_token
from django.views import generic
from django.http import HttpResponseServerError
from PIL import Image
from .forms import *
from .models import *
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@requires_csrf_token
def custom_server_error(request, template_name='500.html'):
    import requests
    import json
    import traceback
    requests.post(
        'https://hooks.slack.com/services/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',
        data=json.dumps({
            'text': '\n'.join([
                f'Request uri: {request.build_absolute_uri()}',
                traceback.format_exc(),
            ]),
            'username': 'Django エラー通知',
            'icon_emoji': ':jack_o_lantern:',
        })
    )
    return HttpResponseServerError('<h1>Internal Server Error</h1>')


def index(request):
    """トップページ"""
    params = {
        "picture_form":UploadPictureForm(),
        "serch_form":SerchClassForm()
    }
    # register_classification()
    return render(request, "garbage/index.html", params)

"""
# csvからデータベースにデータ登録用
def register_classification():
    print(BASE_DIR)
    file = BASE_DIR + "/ML_model/classification.csv"
    with open(file, "r") as f:
        data = f.read().split("\n")[:-1]
        for datum in data:
            key, classification = datum.rsplit(",", 1)
            record = Classification(key=key, classification=classification)
            record.save()
"""

def result(request, num=0):
    """結果表示画面"""
    if num:
        img = f"{BASE_DIR}/static/garbage/media/images/temp{num}.jpg"
    else:
        form = UploadPictureForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data["img"]
        else:
            return index(None)

    pred = predict(img)
    params = {
        "img":img,
        "pred":pred,
        "serch_form":SerchClassForm()
    }
    return render(request, "garbage/result.html", params)


def search(request):
    print(request.GET["word"])
    results = Classification.objects.filter(key__contains=request.GET["word"])
    find = False if len(results)==0 else True
        
    params = {
        "serch_results": results,
        "serch_form":SerchClassForm(),
        "find":find
    }
    return render(request, "garbage/search.html", params)

def opinion(request): 
    params = {
        "redirect":False,
        "serch_form":SerchClassForm(),
        "opinion_form":OpinionForm()
    }
    return render(request, "garbage/opinion.html", params)

def opinion_submit(request):
    opinion_form = OpinionForm()
    opinion_form.title = request.POST["title"]
    opinion_form.text = request.POST["text"]
    opinion_form.send_email()

    params = {
        "redirect":True,
        "serch_form":SerchClassForm(),
        "opinion_form":OpinionForm()
    }
    return render(request, "garbage/opinion.html", params)


def predict(img):
    """予測モデル"""
    # 読み込み
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.preprocessing import image
    from keras.models import model_from_json
    from PIL import Image

    model = model_from_json(open(BASE_DIR + "/model/model.json").read())
    model.load_weights(BASE_DIR + '/model/param.hdf5')

    img_width, img_height = 150, 150
    img = Image.open(img)
    img.save(BASE_DIR + "/static/garbage/media/images/image.jpg")
    img.save(BASE_DIR + "/staticfiles/garbage/media/images/image.jpg")
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
    pred_list = [[c, s, d] for (c, s, d) in zip(classes, pred*100, days)]
    pred_list = sorted(pred_list, key=lambda x:x[1], reverse=True)
    pred_list = [[c, "{:.2f}".format(s), d] for (c, s, d) in pred_list]
    return pred_list
