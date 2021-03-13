
## 初版の作戦

- train.csv/test.csv
  - 正解
    - like: 作品につけられたいいねの数。0以上の整数。この値が予測対象です。
  - 使用中
    - カテゴリ変数として
      - principal_maker ... 製作者。相関はあると思うが、testにはtrainにはいない製作者がいる模様。これにどう対処するか。
      - principal_or_first_maker ... principal_makerがあればいらない気もする。
      - copyright_holder ... 権利保持者。結構少ないカテゴリ。しかしこれもtrainにないやつがある。
      - acquisition_method ... tr/ttで重複しているので使える。NaNはunknownにすべきだろう。カテゴリ変数。
      - acquisition_credit_line ... 誰から誰へのgift化などの情報。
      - dating_year_early: 製作期間の始まり・年
      - dating_year_late: 製作期間のおわり・年
      - dating_period: 制作のされた世紀。開始と終了があればいらない気がする。
    - 数値として
      - dating_interval ... 製作期間の長さ
  - 未使用
    - object_id ... 単なる識別子(重複なし)
    - art_series_id ... 同じシリーズを示すID。train/testに重複はない。
    - acquisition_date ... 収集したタイムスタンプ。年月くらいは使ってカテゴリ変数にするか。
    - title, desc, long_title, more title ... likeと相関の高い単語があれば有効かもしれない。
      - BERT ... オランダ語に対応していないのでムズイ。google翻訳で一旦英語に変換するか？
      - Word2Vec ... 単純な空白区切りで学習する。
    - sub_title ... サイズ情報だが、h,w,t,dとあり、t,dの違いが判らん。使えなくはないかも
  - 使わない気がしているもの
    - dating_presenting_date ... 製作期間。使わない。後述の情報があれば不要そう。  
    - dating_sorting_date: 並び替え表示の際に用いられる代表年。始まりと同じだからいらないんじゃね？

## ver.2

### 分析

- train.csv/test.csv
  - ★タイトルの言語情報を使う。
    - 訪問者が英語メインであることを考えると、英語がlikesの傾向が高くなるのは間違いなさそうだが。。。
    - https://www.guruguru.science/competitions/16/discussions/f463dac2-4233-42d2-8629-ca99a9689987/
    - windowsにpycld2やfasttextをインストールするのが少し難しい…
  - ★タイトルの単純なTFIDF
  - ★sub_titleの使用
    - https://www.guruguru.science/competitions/16/discussions/556029f7-484d-40d4-ad6a-9d86337487e2/
- color.csv
  - 今は使わない。
    - paletteを使えば十分かな。
- historical_person.csv
  - ★歴史的な人物がいるかどうかは重要かも。人物数で数値化する。
- maker.csv
  - 今は使わない。
    - 作家の情報。年数は、制作年があれば十分な気がするのでまずは使わない。
    - 国名はカラムだけでデータがない。生誕場所、没場所もあるけど、関係あるかなぁ？
- materials.csv
  - 素材情報
  - ★カテゴリ変数としていれていいのではないか。
- object_collection.csv
  - 形式
  - ★カテゴリ変数としていれていいのではないか。
- paltette.csv
  - 色情報
  - 複数の色があるので、特徴量は色々考えられる。
    - ★全体的な色味
    - ★輝度分散、色差分散、RGB分散くらいかなぁ
- principal_maker
- principal_maker_occupation
  - ★関わった人数の数くらいは使っていい
  - あとは作家をベクトル化した後には使えるけど、時間がかかるので今はいいかな。
  - ベクトルかはTargetEncodingでもいいか。
- production_place
  - 形式
  - ★カテゴリ変数としていれていいのではないか。
- その他
  - Catboost(HighCardinalityのため有効化かも)
  - TargetEncoding

### 順序
- CV/LB乖離の対策
  - 済：StraightKFold, GroupKFoldを使う。
    - https://www.guruguru.science/competitions/16/discussions/092c2925-6a63-4e65-8057-6ea50fc660dd/
- モデル
  - 済：Catboostの使用
- 追加情報の使用
  - 済：historical_person.csv
    - 歴史的な人物がいるかどうかは重要かも。人物数で数値化する。
  - 済materials.csv/object_collection.csv
    - 済：工夫なしでカテゴリ変数として追加
  - production_placeは国名に変更
    - https://www.guruguru.science/competitions/16/discussions/970ced6d-f974-4979-8f04-dbcf1c2f51a0/
  - principal_maker
    - 人数のみ
  - 済：paltette.csv
    - 済：全体の色
    - 済：輝度分散、色差分散、RGB分散
  - タイトル
    - 済：タイトルの言語判定結果
    - TFIDF
  - 済：sub_titleの使用
    - 済：https://www.guruguru.science/competitions/16/discussions/556029f7-484d-40d4-ad6a-9d86337487e2/
  - Box-Cox