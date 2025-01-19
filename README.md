# 解析after_1 リポジトリ

## プロジェクト概要
このリポジトリは、ドメイン適応事前学習と感情スタンス分析を行うためのコードを提供します。

## 主な機能
- ドメイン適応事前学習
- 感情分析
- スタンス分析

## 使用技術
- Python 3.8+
- PyTorch
- Transformersライブラリ
- scikit-learn

## セットアップ手順

1. リポジトリをクローン:
```bash
git clone https://github.com/saotomem21/-after01.git
cd -after01
```

2. 仮想環境の作成と依存関係のインストール:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## モデルファイルのダウンロード手順

1. Google Driveからモデルファイルをダウンロード:
   - [model.safetensors](https://drive.google.com/drive/folders/1CzMnxfFeepRnZvS5PoIE-h_WbnFZt0U7?usp=drive_link)
   - [model.safetensors.partaa](https://drive.google.com/drive/folders/1CzMnxfFeepRnZvS5PoIE-h_WbnFZt0U7?usp=drive_link)
   - [model.safetensors.partab](https://drive.google.com/drive/folders/1CzMnxfFeepRnZvS5PoIE-h_WbnFZt0U7?usp=drive_link)
   - [model.safetensors.partac](https://drive.google.com/drive/folders/1CzMnxfFeepRnZvS5PoIE-h_WbnFZt0U7?usp=drive_link)
   - [model.safetensors.partad](https://drive.google.com/drive/folders/1CzMnxfFeepRnZvS5PoIE-h_WbnFZt0U7?usp=drive_link)
   - [model.safetensors.partae](https://drive.google.com/drive/folders/1CzMnxfFeepRnZvS5PoIE-h_WbnFZt0U7?usp=drive_link)

2. ダウンロードしたファイルを`domain_adapted_model`ディレクトリに配置

3. 以下のコマンドでファイルを結合:
   ```bash
   cat domain_adapted_model/model.safetensors.part* > domain_adapted_model/model.safetensors
   ```

## リポジトリの使用方法

```bash
# 依存関係のインストール
pip install -r requirements.txt

# スクリプトの実行例
python domain_adaptive_pretrain.py
python sentiment_stance_analysis.py
