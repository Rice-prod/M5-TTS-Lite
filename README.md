# M5-TTS-Lite

面向 AX650N 的中英文轻量级流式语音合成研究项目，基于 [M5-TTS](https://github.com/ASLP-lab/M5-TTS)。当前工程用真实录音做 Teacher forcing，只训练裁剪后的 Depth Decoder，冻结 Temporal Backbone、Codec、嵌入、投影和音频输出头。

**首次运行顺序**：安装环境 → 下载 Teacher / Codec → 整理语料 → 提取轨迹 → 蒸馏训练 → 对比合成 → 导出 ONNX。建议先用少量录音跑通，再用 1h、3h、10h 子集比较效果。

## 1. 安装环境

以下命令均在 **M5-TTS-Lite 仓库根目录**执行。推荐 Linux x86_64、Python 3.10。正式提轨与训练建议使用 CUDA。当前实现使用 SDPA。

```bash
mkdir -p third_party data outputs
git clone https://github.com/ASLP-lab/M5-TTS.git third_party/M5-TTS
git -C third_party/M5-TTS checkout b6050656c425a994e604f6e46c884e869f54261c

python -m venv .venv
source .venv/bin/activate
python -m pip install -r third_party/M5-TTS/requirements.txt

export M5_UPSTREAM="$PWD/third_party/M5-TTS"
export HF_HUB_CACHE="$PWD/outputs/hf-cache"
export M5_TN_CACHE="$PWD/outputs/text-normalizer-cache"

python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m unittest discover -s tests -v
```

测试覆盖码本预测顺序、KV Cache 对齐、裁剪权重迁移、训练冻结范围、权重回载和数据隔离。用 CUDA 时确认上面的 PyTorch 检查返回 `True`。纯 CPU 环境按[运行手册的 CPU 安装说明](docs/distillation.md#1-安装与固定基线)安装，再将提轨和训练命令中的 `--device cuda` 改为 `--device cpu`。新终端需要重新激活环境并设置以上变量。

## 2. 下载并固定 Teacher 与 Codec

人声克隆流程使用上游 VoiceClone 权重作为 Teacher。复现正式实验时，加上对应的 `--revision <commit>` 固定模型版本。

```bash
hf download ASLP-lab/M5-TTS-50M-VoiceClone --local-dir weights/teacher \
  --include 'model.safetensors' 'config.json' 'conditioning.json' 'tokenizer/*'
hf download ASLP-lab/M5-TTS-Audio-Tokenizer --local-dir weights/codec \
  --include '*.json' '*.safetensors'
```

确认 `weights/teacher/` 含模型权重、模型配置、conditioning 配置和 `tokenizer/`，`weights/codec/` 含模型权重和预处理配置。训练集与验证集应使用同一套权重和 Codec。原始语料、参考音频、权重及实验产物放在 Git 忽略目录，使用前核查授权。

## 3. 整理语料并生成清单

把已分句、转写准确且有使用授权的录音放入 `data/raw/`，在 `data/corpus.csv` 中每行登记一条。下面是格式示例，路径和转写应替换成自己的数据。

```csv
utt,audio,text,lang,speaker,group,split,license
zh001,raw/zh001.wav,请打开蓝牙。,zh,speaker01,session01,train,licenses/zh.txt
en001,raw/en001.wav,Please open the window.,en,speaker02,session02,train,licenses/en.txt
mix001,raw/mix001.wav,请连接 Wi-Fi。,mix,speaker03,session03,train,licenses/mix.txt
zh002,raw/zh002.wav,请调低音量。,zh,speaker04,session04,val,licenses/zh.txt
en002,raw/en002.wav,The meeting starts at nine.,en,speaker05,session05,test,licenses/en.txt
```

- `utt` 全局唯一。`audio` 的相对路径以 CSV 所在目录为基准。文本含英文逗号时，用双引号包住整个字段。
- `lang` 支持 `zh`、`en`、`mix`。提轨时混读采用上游的中文 conditioning。
- `speaker` 是全局说话人 ID，`group` 是原始录音或会话 ID。给同一说话人或录音组分配同一个 `split`。训练、验证必须各有样本，最终评估使用独立 `test` 集。
- `license` 填许可记录标识。先完成转写核验和近重复检查。

```bash
python -m m5_lite.data data/corpus.csv data/manifests --seed 42
cat data/manifests/duration_seconds.json
```

产物包括 `train.jsonl`、`val.jsonl`、`test.jsonl` 及 `train_1h.jsonl`、`train_3h.jsonl`、`train_10h.jsonl`。子集按中文 60%、英文 30%、混读 10% 的目标时长选择整句。语料不足时使用对应语言的全部可用样本，实际时长以 `duration_seconds.json` 为准。

## 4. 提取 Teacher 轨迹

先用 `train_1h` 和 `val`。下面两条命令分别生成训练和验证轨迹。

```bash
python -m m5_lite.tracks --upstream "$M5_UPSTREAM" \
  --teacher weights/teacher --codec weights/codec \
  --manifest data/manifests/train_1h.jsonl --output tracks/train_1h \
  --topk 16 --temperature 2 --device cuda

python -m m5_lite.tracks --upstream "$M5_UPSTREAM" \
  --teacher weights/teacher --codec weights/codec \
  --manifest data/manifests/val.jsonl --output tracks/val \
  --topk 16 --temperature 2 --device cuda
```

脚本将录音转为 24 kHz 单声道，用 Codec 提取 16 个码本，再用真实码本前缀计算 Teacher 的 CB1–CB15 预测分布，排除 EOS 帧。成功时会打印帧数和分片数。确认两个输出目录均有 `tracks.json`、`utterances.jsonl` 和 `*.safetensors`，且帧数大于零。Top-K 或温度调整后，需要重提训练和验证轨迹。

## 5. 裁剪并训练 Student

默认配置在 [configs/depth_s.json](configs/depth_s.json)。Student 从 Teacher 的第 0、2、4 层迁移权重，FFN 中间宽度裁到 1024。训练用 Teacher 的 Top-K 和尾部概率算 KL，同时用真实码本算 CE。

```bash
python -m m5_lite.train --upstream "$M5_UPSTREAM" --teacher weights/teacher \
  --train tracks/train_1h --val tracks/val --config configs/depth_s.json \
  --output runs/depth_s --device cuda
```

- 通路检查可复制配置到 `outputs/` 并将 `epochs` 改成 1。显存不足时调小 `batch_frames`。做 CE-only 对照时，将 `kd_weight` 设为 0，并使用新的实验目录。
- `runs/depth_s/epoch-0000/` 是裁剪后、训练前的基线。其余 `epoch-NNNN/` 保存模型、Tokenizer、优化器状态及 `distill.json`。查看验证 `loss`、`kl`、`ce` 及 `is_best`，选出候选轮次做试听和端到端对比。完整 FP32 bundle 可先按每轮约 600 MB 预留空间。
- **续训**时保留原训练和验证轨迹及训练设置，将配置里的 `epochs` 改为新的目标总轮数，再执行原命令并追加 `--resume runs/depth_s/epoch-0001`。将路径换成实际要恢复的轮次，并确保后续轮次的输出目录可用。
- 扩到 3h 或 10h 时，仅替换训练清单，重新提取对应训练轨迹。使用同一验证轨迹，并为每个实验指定独立的运行目录。

## 6. 用相同输入对比 Teacher 和 Student

准备参考音频和对应的准确转写。以下使用上游示例参考音频。将 `STUDENT` 改为上一步选出的轮次。

```bash
STUDENT=runs/depth_s/epoch-0001
PROMPT_WAV="$M5_UPSTREAM/assets/aslp_default_voice.wav"
PROMPT_TEXT='猪携带该疾病，然后通过蚊子传染给人类。'
TARGET_TEXT='你好，欢迎使用语音助手。'

for model in teacher student; do
  MODEL_DIR=weights/teacher
  if [ "$model" = student ]; then MODEL_DIR="$STUDENT"; fi
  python -m m5_lite.infer --upstream "$M5_UPSTREAM" --model "$MODEL_DIR" \
    --codec weights/codec --device cpu --threads 1 --seed 42 \
    --prompt-wav "$PROMPT_WAV" --prompt-text "$PROMPT_TEXT" \
    --text "$TARGET_TEXT" --lang Chinese --output "outputs/$model.wav" || break
done
```

每个模型都生成 WAV 和同名的 `.stats.json`。试听漏字、重复、停顿和音色，记录有没有达到最大生成长度的警告，再比较音频时长和 RTF。RTF 以模型加载后的推理耗时除以输出音频时长，含参考音频处理。英文对比使用 `--lang English`，并替换为对应的测试文本和参考录音。正式质量对比使用同一批独立测试样本，分别记录中文 CER 和英文 WER。

## 7. 导出 ONNX 并验证 CPU 流式推理

沿用上一步的环境变量、Student 路径和参考音频。导出会运行上游的 ONNX Runtime 校验。

```bash
python -m m5_lite.export_onnx --upstream "$M5_UPSTREAM" --model "$STUDENT" \
  --codec weights/codec --output outputs/student_onnx

python "$M5_UPSTREAM/infer_onnx.py" --model outputs/student_onnx \
  --device cpu --threads 1 --stream --verify \
  --prompt-wav "$PROMPT_WAV" --prompt-text "$PROMPT_TEXT" \
  --text "$TARGET_TEXT" --lang Chinese --output outputs/student_onnx.wav
```

确认输出目录包含模型和元数据，且推理生成有效 WAV。查看统计中的 `eos`、`truncated`、`rtf` 和 `first_pcm_s`，同时试听输出。此步骤验证 ONNX Runtime 路径。Pulsar2 量化、AX650N C++ 接入、板端性能验收及自由运行轨迹仍按[总体实施方案](docs/01_蒸馏总体实施方案.md)推进。

## 常见问题

| 现象 | 检查与处理 |
| --- | --- |
| 报上游版本不匹配 | 运行 `git -C "$M5_UPSTREAM" rev-parse HEAD`，确认是第 1 步指定的提交。 |
| 清单生成时报跨集合冲突 | 核对说话人和录音组 ID，把相关样本放到同一集合，再生成到新的目录。 |
| 轨迹元数据不兼容 | 核对 Teacher、Codec、Top-K 和温度，重提不匹配的轨迹。 |
| 输出路径已存在 | 给清单、轨迹、实验或推理产物指定新路径。续训时选择已完成的轮次并从下一个轮次开始写入。 |
| 推理达到长度上限 | 查看日志和试听结果，区分正常长句与未生成 EOS。PyTorch 用 `--max-audio-length-ms` 调整，ONNX 用 `--max-frames` 调整。 |

实现入口见 [m5_lite/](m5_lite/)，回归测试见 [tests/](tests/)，轨迹格式和实验细节见[蒸馏工程运行手册](docs/distillation.md)。
