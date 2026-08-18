# 合成会议分离示例

本示例生成一段带 planted ground truth 的双说话人合成会议 WAV，并用 planted speaker identity 构造 oracle embedding，随后执行真实的聚类、DER 计算与 SVG 渲染。oracle embedding 用于隔离并验证 diarization mechanics，不代表从真实音频训练或评测了说话人模型。

```bash
.venv/bin/python examples/diarization_meeting/run.py build/example-diarization
```

输出包含 `summary.json`、`meeting.wav`、说话人时间线和 DER 分解图。所有数字来自本次确定性合成运行。
