import requests
import lxml.html

session = requests.Session()
response = session.get('https://manage.delight-system.com/threeR/web/bunbetsu?menu=bunbetsu&jichitaiId=kashiwashi&areaId=22125&areaName=%2F&lang=ja&benriCateId=7&bunbetsuCateId=7&faqCateId=%2F&howToCateId=&search=%E3%83%9A%E3%83%83%E3%83%88%E3%83%9C%E3%83%88%E3%83%AB&dummy=')

root = lxml.html.fromstring(response.content)
stuff_list = root.cssselect('.panel-heading')
dic_class = {}
for stuff in stuff_list:
    key = stuff.cssselect('a')[0].text
    classification = stuff.cssselect('div div.panel-body')[0].text
    dic_class[key] = classification

with open("classification.csv", "w") as f:
    for key, value in dic_class.items():
        f.write(f"{key},{value}\n")

