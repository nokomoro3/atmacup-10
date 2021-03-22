# atmacup-10

atmaCup #10コンペ用レポジトリ

## コンペのページ
- https://www.guruguru.science/competitions/16/

## solutionについての解説
- https://www.guruguru.science/competitions/16/discussions/71f8aca0-9b16-4d5f-942b-62db9acd5055/

## 環境設定

[INSTALL.md](./INSTALL.md)を参照ください。

## 実行方法

```sh
(.venv) $ python3 main.py
```

## 最終スコア

- Public: 0.9569 (28th)
- Private: 0.9801 (27th)

## CV/LBの遷移

|fileName|CV|LB|desc|
|:---|:---|:---|:---|
|submission_20th   | 忘却 | 0.9590 | optunaで調整 |
|submission_19th   | - | - | バグ |
|submission_18th   | 0.9990 | 0.9569 | w2v追加 |
|submission_17.2th | 0.9916 | 0.9569 | category化修正漏れ |
|submission_17th   | 0.9908 | 0.9582 | maker情報追加 |
|submission_16th   | 0.9930 | 0.9590 | 場所情報追加 |
|submission_15th   | 0.9901 | 0.9630 | 不明ラベルの共通カテゴリ化 |
|submission_14.2th | 0.9953 | 0.9646 | LGMBパラメタ調整2 |
|submission_14.1th | 1.0004 | 0.9639 | LGMBパラメタ調整1 |
|submission_14th   | 1.0004 | 0.9660 | color統計情報追加。<br>palette見直し。<br>収集日を数値化特徴量として使用。|
|submission_13th   | 1.0051 | 0.9763 | word2vecを追加|
|submission_12th   | 1.0032 | 0.9723 | 講座#2を参考に、TDIDF追加。|
|submission_11th   | 1.0374 | 1.0445 | 構成変更。<br>講座#2を参考に、countEncoding追加。<br>pallet計算の特徴量を追加。|
|submission_10th   | 1.0546 | 1.0445 | technique, object_collection情報をone-hotで追加。<br>講座#1を参考に文字列長とCountEncodingを追加。 |
|submission_9th    | 1.0705 | 1.0480 | 言語情報をdesc, more_titleなどに拡大<br>material情報をone-hotで追加。 |
|submission_8th    | 1.0903 | 1.0608 | size追加 |
|submission_7th    | 1.1193 | 1.0949 | titleの言語情報使用 |
|submission_6th    | 1.1407 | 1.1205 | パレット統計情報使用 |
|submission_5th    | 1.2228 | 1.2081 | catboost使用 |
|submission_4th    | 1.2383 | 1.2334 | CV/LBの乖離修正 |
|submission_3rd    | 1.1409 | 1.2396 | CV/LBの乖離修正(整数化にバグ) |
|submission_2nd    | 1.2418 | 1.2018 | ベースライン |
