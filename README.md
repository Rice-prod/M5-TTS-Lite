# M5-TTS-Lite

面向 AX650N 的中英文轻量级流式语音合成研究项目，基于 [M5-TTS](https://github.com/ASLP-lab/M5-TTS)。

## 研发方案

[完整小样本蒸馏实施方案（V1.0）](docs/01_蒸馏总体实施方案.md)

## 核心路线

- 冻结 Temporal Backbone 与 Codec，优先蒸馏 Depth Decoder。
- 先做随机权重上板测速，再按 1h → 3h → 10h 的嵌套双语数据规模开展蒸馏；通过质量与速度门禁即停止扩充。
- 独立评估 Host 音频头低秩化，随后试验分组预测。
- 在 LITTLE 单核、OMP=1 约束下，以板端实测 RTF、首包时延和中文 CER / 英文 WER 为验收依据。

## 工程目录

| 目录 | 用途 |
| --- | --- |
| docs/ | 研发方案与后续实验文档 |
| configs/ | 训练及编译配置 |
| datasets/ | 数据处理脚本（不存放原始语料） |
| models/ | Student 网络结构 |
| distill/ | 蒸馏训练及 Teacher 轨迹提取 |
| export/ | ONNX 导出 |
| compile/ | Pulsar2 量化编译 |
| inference/ | Python / C++ 推理 |
| evaluation/ | 中英文音质评测 |
| benchmarks/ | 板端性能测试 |
| scripts/ | 工程运行脚本 |

## 当前状态

方案已提交；代码目录仅为占位，尚未实现或验证训练、编译和推理代码。目标指标不是已实现的结果。

原始语料、参考音色、Teacher 轨迹、模型权重及本地实验产物不纳入版本控制。使用前核查上游模型和数据许可。
