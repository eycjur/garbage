# 画像でゴミ分類！

機械学習を用いて画像分類を行うことで画像からゴミの分類を調べるというアプリです。

## 製作動機
私自身が普段、ゴミを適当に捨てているなという実感があったので、もっと手軽に分別を調べられるアプリがあれば便利だなと思い一人で作成しました。

## 利用技術
使用した言語は、バックエンドはDjango(python)、フロントエンドはHTML,CSS（フレームワークはBootstrap）,JavaScriptです。  
機械学習のモデルについては、データセットは自分で撮影した画像を用いており、モデルはkerasでVGG16をファインチューニングして作成しました。

## 工夫した点
工夫した点はアウトプットすることです。このアプリを作るにあたっては悩んだ点や苦労した点を記録しておくとともに言語化できるように毎日Qiitaに記事を投稿することを自分に課しました。これにより自分の理解度を確認しながら頭を整理することができました。

## 苦労した点
苦労した点は画像の処理についてです。フォームで読み込んだ際の画像をどのようにバックエンド側に渡せばいいのか悩みましたが、いろいろ実験してみることで解決策を見つけ出すことができました。  

## デモ・説明用動画
- [「画像でゴミ分類！」アプリ解説](https://www.youtube.com/watch?v=P8194a3Lhac)

## Qiita記事
- [「画像でゴミ分類！」アプリ作成日誌day1～データセットの作成～](https://qiita.com/eycjur/items/7d8223b28758c7dfaaa0)
- [「画像でゴミ分類！」アプリ作成日誌day2～VGG16でFine-tuning～](https://qiita.com/eycjur/items/3e954cb70dc15f996c2d)
- [「画像でゴミ分類！」アプリ作成日誌day3～Djangoでwebアプリ化～](https://qiita.com/eycjur/items/9c618538177c82f7fdc3)
- [「画像でゴミ分類！」アプリ作成日誌day4～Bootstrapでフロントエンドを整える～](https://qiita.com/eycjur/items/7b58c28eb8b16e722b5d)
- [「画像でゴミ分類！」アプリ作成日誌day5～Bootstrapでフロントエンドを整える2～](https://qiita.com/eycjur/items/08808097acd625e00652)
- [「画像でゴミ分類！」アプリ作成日誌day6～ディレクトリ構成の修正～](https://qiita.com/eycjur/items/f436787c2d517af6cad6)
- [「画像でゴミ分類！」アプリ作成日誌day7～サイドバーのスライドメニュー化～](https://qiita.com/eycjur/items/009dca5866c3e9ad0dd6)
- [「画像でゴミ分類！」アプリ作成日誌day8～herokuデプロイ～](https://qiita.com/eycjur/items/ebfae5db5dd1cd8349ea)


