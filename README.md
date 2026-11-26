# M5-TTS-Lite

面向 AX650N 的中英文轻量级流式语音合成研究项目，基于 [M5-TTS](https://github.com/ASLP-lab/M5-TTS)。

[蒸馏工程运行手册](docs/distillation.md) · [小样本蒸馏总体实施方案](docs/01_蒸馏总体实施方案.md)

## 蒸馏流程

1. 检查语料 CSV，按说话人和录音组隔离数据，输出嵌套的 1h、3h、10h 中英混合训练清单。
2. 用同源 Codec 编码音频，以真实码本前缀提取 Teacher 的 Temporal hidden 和 CB1–CB15 条件分布。
3. 从 Teacher 迁移三层 Depth Decoder，裁剪 SwiGLU 中间通道，冻结其余参数，使用 Top-K + 尾部 KL 和真实码本 CE 训练。
4. 保存可续训的 Student bundle；执行 PyTorch WAV 推理，或导出上游 ONNX Runtime 模型。

| 路径 | 内容 |
| --- | --- |
| `m5_lite/data.py` | 语料检查、切分清单和实际小时数 |
| `m5_lite/tracks.py` | Teacher 轨迹提取与二进制分片 |
| `m5_lite/depth.py` | 层及 FFN 通道迁移、全帧 Depth 前向 |
| `m5_lite/train.py` | KD/CE、验证、权重保存及续训 |
| `m5_lite/infer.py`、`m5_lite/export_onnx.py` | WAV 推理、RTF 记录、ONNX 导出 |
| `configs/depth_s.json`、`tests/` | 默认配置与回归测试 |

当前工程实现的是基于真实录音的 teacher-forced Depth 蒸馏。自由运行轨迹、Host 低秩头、分组预测、Pulsar2 量化和 AX650N C++ 集成沿用总体方案的实验计划。板端 RTF、首包时延和中英文识别错误率须用同源模型实测。

原始语料、参考音色、Teacher 轨迹、模型权重及本地实验产物存放在本地忽略目录中。使用前核查上游模型及语料授权。
