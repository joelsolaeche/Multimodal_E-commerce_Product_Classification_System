# マルチモーダル EC商品分類システム

本プロジェクトは、**テキスト + 画像** を用いたマルチモーダルの商品分類システムを、Eコマース向けに実装したものです。

---

## 🆕 実データ対応

本システムは **Railway 上の PostgreSQL** を用いた実データをサポートしています。詳細は [実データデプロイメントガイド](DEPLOYMENT_GUIDE.md) を参照してください。

### 主な特徴
- PostgreSQL データベースによる商品データ管理  
- 大規模埋め込みファイル（テキスト・画像）のサポート  
- データベースが利用できない場合の自動モックデータ切り替え  
- Railway 上でスケーラブルに動作するアーキテクチャ  

---

## プロジェクト構成

- `api_server.py`: FastAPIベースの推論サーバー  
- `src/`: 機械学習モデルの実装  
- `data/`: 商品データと画像  
- `Embeddings/`: 事前計算されたテキスト・画像埋め込み  
- `multimodal-ecommerce-demo/`: Next.jsフロントエンド  
- `import_data_to_postgres.py`: PostgreSQLへデータを取り込むスクリプト  

---

## セットアップ手順

### 1. ローカル開発環境
```bash
# 依存関係をインストール
pip install -r requirements-api.txt

# APIサーバーを起動
python api_server.py
```

### 2. データベース設定
```bash
# PostgreSQL接続文字列を設定
export DATABASE_URL="postgresql://postgres:password@localhost:5432/postgres"

# データをPostgreSQLにインポート
python import_data_to_postgres.py
```

### 3. Railway デプロイ
Railway + PostgreSQL を用いたデプロイ手順は [Deployment Guide](DEPLOYMENT_GUIDE.md) を参照してください。

---

## API エンドポイント

- `/api/classify/text`: テキストベースの商品分類  
- `/api/classify/image`: 画像ベースの商品分類  
- `/api/classify/multimodal`: テキスト + 画像のマルチモーダル分類  
- `/api/products`: 商品リスト取得  
- `/health`: APIのヘルスチェック  

---

## フロントエンドデモ

Next.js によるデモで、マルチモーダル分類機能を確認できます。

```bash
# フロントエンド起動
cd multimodal-ecommerce-demo
npm install
npm run dev
```

---

## データ管理オプション

1. **PostgreSQL on Railway（推奨）**  
   - 商品データ（約18MB）とサンプル埋め込みを格納  
   - 無料枠で1GBストレージ  
   - 自動バックアップとスケーリング  

2. **Google Cloud Storage**  
   - 大規模埋め込みファイル（800MB以上）を保存  
   - オンデマンド、または起動時にダウンロード可能  
   - 無料枠で5GBストレージ  

3. **ハイブリッド方式**  
   - 商品データをPostgreSQLに保存  
   - 埋め込みをクラウドストレージに保存  
   - キャッシュによるパフォーマンス最適化  

---

## パフォーマンスの考慮事項

- Railway 無料枠: 1GB PostgreSQL ストレージ  
- 推奨方法: 商品データ + サンプル埋め込みのみ保存  
- 完全な埋め込みを利用する場合: クラウドストレージを使用、または有料プランにアップグレード  

---

## ライセンス

MIT ライセンス
