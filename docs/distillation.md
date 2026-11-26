# Depth Decoder 蒸馏工程

## 1. 安装与固定基线

在 M5-TTS-Lite 仓库根目录运行。推荐 Linux x86_64、Python 3.10；正式训练使用 CUDA，本流程通过上游 SDPA 支持 CPU 检查。

```bash
git clone https://github.com/ASLP-lab/M5-TTS.git third_party/M5-TTS
git -C third_party/M5-TTS checkout b6050656c425a994e604f6e46c884e869f54261c
python -m venv .venv
source .venv/bin/activate
python -m pip install -r third_party/M5-TTS/requirements.txt
export HF_HUB_CACHE="$PWD/outputs/hf-cache"
export M5_TN_CACHE="$PWD/outputs/text-normalizer-cache"
```

纯 CPU 环境可先从 PyTorch CPU wheel 源安装 `torch==2.6.0`、`torchaudio==2.6.0`，然后将上游依赖清单中的 `torch==2.6.0+cu124`、`onnxruntime-gpu==1.23.2` 分别改为 `torch==2.6.0`、`onnxruntime==1.23.2`，保存到本地 `outputs/` 再安装。SDPA 无需 FlashAttention。

准备本地模型和 Codec 目录。Teacher 目录需包含 `model.safetensors`、`config.json`、`conditioning.json` 和 `tokenizer/`；Codec 需含模型与预处理配置。部署相关实验使用现有部署权重的同源浮点版本。若只需检查工程，可以下载上游 VoiceClone checkpoint 和原 Codec。

```bash
hf download ASLP-lab/M5-TTS-50M-VoiceClone --local-dir weights/teacher \
  --include 'model.safetensors' 'config.json' 'conditioning.json' 'tokenizer/*'
hf download ASLP-lab/M5-TTS-Audio-Tokenizer --local-dir weights/codec \
  --include '*.json' '*.safetensors'
```

正式实验建议在下载时用 `--revision` 指定已记录的模型版本。轨迹元数据会保存实际文件的 SHA-256。

## 2. 数据清单

把已切分、转写、审核的语料登记在 `data/corpus.csv`。相对音频路径以 CSV 所在目录为基准。

```csv
utt,audio,text,lang,speaker,group,split,license
zh001,raw/zh001.flac,请打开蓝牙。,zh,aishell_s001,session_001,train,licenses/aishell3.txt
en001,raw/en001.wav,Please open the window.,en,libri_s001,session_002,val,licenses/libritts.txt
mix001,raw/mix001.wav,请连接 Wi-Fi。,mix,own_s001,session_003,test,licenses/internal.txt
```

- `lang` 使用 `zh`、`en` 或 `mix`。当前提轨把混读映射到上游 `Chinese` conditioning。
- `speaker` 用全局唯一说话人 ID，`group` 用原录音或会话 ID，两者均用于防止跨集合泄漏。可用可选列 `conditioning_speaker`、`accent`、`caption` 设置模型 conditioning。
- `license` 填写来源的许可记录标识。输入文本作为转写传入上游训练 tokenizer。抽样前完成训练文本规范、转写核验和近重复审核。
- `train_1h`、`train_3h`、`train_10h` 各自按中文 60%、英文 30%、混读 10% 的目标时长追加整句。配额不足时使用该语言全部样本，最后一句可使目标时长略微超额。以 `duration_seconds.json` 为准。

```bash
python -m m5_lite.data data/corpus.csv data/manifests --seed 42
```

## 3. 缓存 Teacher 轨迹

```bash
for split in train_1h val; do
  python -m m5_lite.tracks --upstream third_party/M5-TTS \
    --teacher weights/teacher --codec weights/codec \
    --manifest "data/manifests/${split}.jsonl" --output "tracks/${split}" \
    --topk 16 --temperature 2 --device cuda || break
done
```

提轨将录音转为 24 kHz 单声道，按上游 Codec 协议取原始 16 码本，使用 `+1` 后的模型 ID。对目标帧，Temporal 输入止于上一帧；Depth 序列首项是该帧的 Temporal hidden，后接 CB0 至 CB14 的 embedding。Depth 的第 1–15 个位置预测 CB1–CB15，并排除 EOS 帧。缓存由 Safetensors 分片、版本与哈希元数据及 utterance 索引组成。提轨温度写入缓存，调整温度后需要重新提轨。当前缓存使用 FP32 hidden 和概率，约 3.1 KB/帧（Top-16、hidden 512）。

## 4. 裁剪、训练与续训

```bash
python -m m5_lite.train --upstream third_party/M5-TTS --teacher weights/teacher \
  --train tracks/train_1h --val tracks/val --config configs/depth_s.json \
  --output runs/depth_s --device cuda
```

默认从第 0、2、4 层初始化，保持 hidden、attention 头和原音频输出头。FFN 默认 1024，按门控、上投影和下投影的联合权重范数选择共享通道。同一架构的 CE-only 对照可把配置的 `kd_weight` 设为 0。训练使用单设备 FP32 AdamW；显存不足时先调低 `batch_frames`。

`epoch-0000` 是裁剪初始化的对照。每轮保存完整的 FP32 推理权重，单个 bundle 的磁盘预算可先按 600 MB 估计。训练输出 `loss`、`kl`、`ce`，并保存 `distill.json`、权重、Tokenizer 和优化器状态。用 `--resume runs/depth_s/epoch-0001` 可从已完成轮次继续；配置中的 `epochs` 为目标总轮数。根据验证日志选出候选，并用固定文本、参考音频、采样参数比较 Teacher 和 Student 的真实生成结果。使用 3h 或 10h 子集时，沿用同一验证清单，输出到新的实验目录。

## 5. 合成与 ONNX

```bash
STUDENT=runs/depth_s/epoch-0001
python -m m5_lite.infer --upstream third_party/M5-TTS --model "$STUDENT" \
  --codec weights/codec --device cpu --threads 1 --seed 42 \
  --prompt-wav data/reference.wav --prompt-text '这里填写参考录音的准确转写。' \
  --text '你好，欢迎使用语音助手。' --lang Chinese --output outputs/student.wav
```

把 `--model` 指向 `weights/teacher` 并使用另一个输出文件，可用相同输入做基线比较。CustomVoice checkpoint 使用 `--speaker-id` 及其支持的预置音色。统计 JSON 中的 RTF 以本次完整推理耗时除以实际输出音频时长，包含参考处理且从模型加载完成后开始计时。

```bash
python -m m5_lite.export_onnx --upstream third_party/M5-TTS --model "$STUDENT" \
  --codec weights/codec --output outputs/student_onnx
python third_party/M5-TTS/infer_onnx.py --model outputs/student_onnx \
  --device cpu --threads 1 --stream \
  --prompt-wav data/reference.wav --prompt-text '这里填写参考录音的准确转写。' \
  --text '你好，欢迎使用语音助手。' --lang Chinese --output outputs/student_onnx.wav
```

ONNX 封装使用上游完整 DepthFrame 图及其 ORT 校验。AX650N 需要根据同源的原板端导出、编译配置和 C++ 缓冲区布局适配独立 prefill/step 图。板端首包时延和 CER/WER 的验收口径见总体实施方案。

## 6. 回归检查

```bash
M5_UPSTREAM=third_party/M5-TTS python -m unittest discover -s tests -v
```

测试包含码本预测位置与逐步 KV 路径的一致性、全宽迁移、Top-K 尾部概率、冻结参数和 checkpoint 回载，以及说话人切分隔离。
