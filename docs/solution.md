みなさまお疲れ様でした。
コンペ初参加でしたが、いろんな方のディスカッションのおかげで
この順位まで上げられました。感謝いたします。

## モデル
lightGBM

## ハイパーパラメータ
- 最初は、@takoi さんのディスカッションそのまま。
  - https://www.guruguru.science/competitions/16/discussions/8d476062-3058-45a3-8a8c-d2d4973862b5/
- 最後にoptunaで最適化したものを使用。（でもあまり性能は変わらず）

## ソースコード
- lightGBMは、@takoi さんの部分を改造して使用、
- 特徴量計算は、@nyk510 さんの入門講座#2を参考に、少しfeatureBlockを改造して使いました。
- notebookに慣れていないので、GithubにコードをUpしておきます。
  - https://github.com/nokomoro3/atmacup-10

## 特徴量について
- 色んな方のdiscussionを参考にしています。
- palette/colorの部分で少し独自性はあるかもしれません。
  - ratioをあえて使わない
  - YUV変換を使う、など

### train.csv/test.csv
- 入門講座#2より
  - https://www.guruguru.science/competitions/16/discussions/95b7f8ec-a741-444f-933a-94c33b9e66be/
    - 数値そのままの特徴量
    - Label Encoding
    - Count Encoding
    - 文字列長
    - TFIDF
- sub_titleによるオブジェクトのサイズ（@skleak さん）
  - https://www.guruguru.science/competitions/16/discussions/556029f7-484d-40d4-ad6a-9d86337487e2/
- 言語判定結果（@Arai-san さん）
  - https://www.guruguru.science/competitions/16/discussions/f463dac2-4233-42d2-8629-ca99a9689987/
- 製作期間（dating_year_early - dating_year_late）（独自というほどのものではない）
- 収集年月 acquisition_date の数値化（独自というほどのものではない）

### historical_person/material/object_collection/technique/production_place/principal_maker
- One Hot Encodingして結合（入門講座#2より）
- それぞれのオブジェクトの合計カテゴリ数も使用（独自というほどのものでは？）
  - より複雑な手法や製作者が絡んでいることが、特徴になるのでは？と考えた。
- Word2Vec（@Arai-san さん）
  - https://www.guruguru.science/competitions/16/discussions/2fafef06-5a26-4d33-b535-a94cc9549ac4/

### palette/color
- ここは少し独自で考えられた。
  - まずRGBからYUV、HSV、YIQに変換
  - それぞれratioで重みづけし、平均、分散、MAX-MIN差を計算
  - 加えてあえてratioで重みづけせず、平均、分散、MAX-MIN差を計算

## 振り返り
- 手一杯でdiscussion投稿まで手が回らず、、、次回から頑張ります！
- Target Encodingなど、Category Encoderを試したかった。（けど時間が足りず...）
  - Category Encodersという便利そうなモジュールがあったので次からはこれを使いたい。
    - https://contrib.scikit-learn.org/category_encoders/
- 特徴量増えてしまったからAdversarial Validationはやりたかった。（けど時間が…）
- 上位の方の解法を聞くに、BERTは試すべきだったな～と感じました。（けど時間…）
- principal_maker_occupation.csvの使い方が思いつきませんでした（どなたか使えましたか？）。
