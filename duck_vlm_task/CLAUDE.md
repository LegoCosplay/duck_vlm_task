# プロジェクト
- microduck（MuJoCo）で、学習済みポリシーの推論＋VLMによるオリジナルタスクを実施する
- タスク：指示されたボールまで歩いて蹴る
- 公式リポジトリは ../microduck_rl（書き換えない）。infer_policy.py は sys.path に ../microduck_rl/scripts を足して import し、PolicyInference と load_bam_model を部品として使う
- 実行：uv run --project ../microduck_rl python ファイル名.py（MuJoCo 3.10.0）
- ステップ：①MjSpecでシーン作成（青いボール）②キーボードなしで前進→停止→キック ③真値のボール位置で比例制御（ベースライン）④ロボット目線の画像 ⑤VLM

# AIへのお願い
- コードは私が書く。完成コードは渡さず、短いヒント・関数名・調べる場所を示し、書いたものをレビューする。教えすぎない
- 日本語で簡潔に