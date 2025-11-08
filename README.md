# transformer-assignment
大模型基础与应用课程期中作业，基于 PyTorch 实现的 Transformer 模型，用于 IWSLT17 数据集上的神经机器翻译任务。


- 训练集：`train.tags.en-de.*`
- 验证集：`dev2010`
- 测试集：`tst2014`

---

## 🚀 快速开始（Windows）

- 训练
```bash
python src\train.py --config configs\transformer.yml
```

- 单句翻译
```bash
python src\translate.py --config configs\transformer.yml --sentence "Eine Gruppe von Menschen steht vor einem Gebäude." --beam_size 5 --length_penalty 0.6
```

- 测试集翻译 + BLEU
```bash
python src\translate.py --config configs\transformer.yml --dataset test --eval --output results\test_pairs.txt --beam_size 5 --length_penalty 0.6
```

生成内容：
- 最优模型：`results/transformer.pt`
- 训练曲线：`results/training_curves.png`
- PPL 曲线：`results/training_ppl.png`
- 翻译对齐：`results/test_pairs.txt`
- BLEU：`results/test_bleu.txt`（开启 `--eval`）

> macOS/Linux 可使用提供的脚本：`scripts/run.sh`

---

## ⚙️ 配置说明（configs/transformer.yml）

关键项：
- 语言：`src_language: 'de'`、`tgt_language: 'en'`
- 分词：`tokenizer: spacy_blank`（推荐，训练推理一致、无需下载模型）
- 长度：`max_seq_len: 60`（含 BOS/EOS；正文约 58）
- 模型：`d_model`、`n_heads`、`d_ff`、`n_encoder_layers`、`n_decoder_layers`、`dropout`
- 训练：`batch_size`、`epochs`、`lr`、`optimizer: AdamW`、`grad_clip`、`seed`、`device`
- 输出：`output_dir`、`model_save_name`、`plot_save_name`
- 速度：`compile: false`（Windows/Triton 不可用时自动回退）

> 你也可以启用注释里的“小模型”以节省资源。

---

## 📊 评估流程

- 数据集翻译并计算 BLEU：
```bash
python src\translate.py --config configs\transformer.yml --dataset test --eval --output results\test_pairs.txt
```

- 结果文件：
  - `results/test_pairs.txt`：原文与模型翻译的配对
  - `results/test_bleu.txt`：sacreBLEU 总分

> 若未安装 `sacrebleu`，脚本会提示安装并跳过评估。

---

## 🧠 模型与实现细节

- 固定正弦位置编码：`src/model.py` 在 `Encoder/Decoder` 中注册 `pos_table` buffer，替代可学习位置嵌入（保留旧变量以兼容已有权重）。
- 注意力稳定性：
  - 掩码填充值使用 `-1e9`（替代 `-inf`），避免整行掩码导致 AMP 下 softmax NaN。
  - 在 `float32` 中进行 softmax 后再转换 dtype。
- DataLoader：Windows 默认 `num_workers=0`，避免多进程问题。
- 词表缓存：`results/vocabs`，按语言对/分词模式/最小频率组织，避免重复构建。

---

## 🪵 目录结构

```text
期中作业
├─ configs/
│  └─ transformer.yml
├─ data/
│  └─ en-de/
├─ results/
│  ├─ transformer.pt
│  ├─ training_curves.png
│  ├─ training_ppl.png
│  └─ vocabs/
├─ scripts/
│  └─ run.sh
└─ src/
   ├─ dataset.py
   ├─ model.py
   ├─ train.py
   └─ translate.py
```

---

## ❓ 常见问题

- 训练出现 `loss=NaN`：
  - 当前实现已修复掩码与 softmax 的数值稳定性。
  - 如仍遇到：降低学习率（如 `5e-5`）、临时关闭 AMP 定位问题、禁用 `fused=True` 的 AdamW（环境较老时）。
- 长度越界：
  - 确保 `max_seq_len <= 512`（`Encoder/Decoder` 的默认构造长度）。
- 分词一致性：
  - 训练与推理统一使用 `spacy_blank`，无需下载 `de_core_news_sm` 或 `en_core_web_sm`。

---

## 📝 许可证

本项目为课程作业示例。若用于开源发布，建议添加 `LICENSE`（例如 MIT）。
