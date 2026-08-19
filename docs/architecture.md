# 架构

choravox 把离线多说话人会议处理拆成可独立验证的纯数据步骤。核心只依赖 NumPy；输入、配置、随机种子和序列化 schema 都显式进入产物。

1. `audio_io`、`resample`、`features` 完成 PCM/WAV、重采样、分帧、log-mel、CMVN 与 delta。
2. `dialogue`、`phonetics`、`synthesis`、`room`、`microphone`、`noise` 生成带 planted words、phones 和 speaker turns 的会议。
3. `vad`、`embeddings`、`diarization`、`overlap` 分别处理语音活动、说话人表示、聚类与重叠帧。
4. `ctc`、`asr`、`alignment` 实现 CTC 路径、SOT speaker-change、speaker attribution 和强制对齐机制。PyTorch 模型是可选层，不属于 NumPy 运行时依赖。
5. `metrics`、`evaluation`、`streaming` 计算 cpWER、DER、JER、条件切片、online/offline gap 与显式 latency。
6. `report` 只消费真实运行产物，生成确定性 SVG、Markdown、HTML 和带摘要的 manifest。

## 可复现边界

默认流程完全离线且只使用确定性合成数据。oracle embedding 或 planted transcript adapter 会在产物中明确标注。真实语料名称只用于机制背景，本项目不下载或报告其指标。

## 机制参考

实现为原创代码，概念背景包括 [Allen 与 Berkley 的 image method](https://doi.org/10.1121/1.382599)、[Dehak 等人的 i-vector/UBM 工作](https://doi.org/10.1109/TASL.2010.2064307)、[Landini 等人的 VBx](https://arxiv.org/abs/2012.14952)、[Kanda 等人的 SOT](https://doi.org/10.21437/Interspeech.2020-1667) 与 [pyannote.audio 3.1](https://arxiv.org/abs/2311.16229)。这些引用解释机制来源，不表示复现论文结果。
