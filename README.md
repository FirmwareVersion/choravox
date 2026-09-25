# choravox

choravox 是一个 Python 3.11+ 离线多说话人会议语音工具包。项目以 NumPy 为核心，在确定性合成会议上实现并验证“谁在何时说了什么”的处理机制。默认流程不下载数据集或预训练权重，也不需要 GPU 或付费服务。

工具链覆盖 WAV/PCM、polyphase resampling、log-mel/CMVN/delta、source-filter speech synthesis、room/microphone/noise simulation、VAD、statistics embedding、AHC/spectral/VB-HMM diarization、overlap detection、CTC/SOT speaker attribution、forced alignment、cpWER/DER/JER、流式处理和确定性报告。PyTorch CPU extra 提供小型可训练 speaker encoder 与 CTC acoustic model。

## 快速开始

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
make test
.venv/bin/choravox --version
```

需要运行可选模型和全部 slow/model 测试时：

```bash
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -e ".[dev,torch]"
make test-all
```

## 离线演示

```bash
.venv/bin/choravox synthesize build/meeting --seed 7
.venv/bin/choravox demo build/report --seed 7
.venv/bin/choravox stream-demo build/stream-report --seed 7
```

`demo` 会执行合成、VAD、planted/oracle diarization 与 transcript adapter、指标计算和 report synthesis，并在输出及 limitations 中说明覆盖事实。它验证整条处理机制，不把 oracle 结果表述为未知音频上的模型精度。

四个可运行示例位于 `examples/`：

- `diarization_meeting`：oracle embedding 聚类、DER 分解与 speaker timeline；
- `asr_overlap_sweep`：CTC/SOT attribution 与 synthetic overlap stress cpWER；
- `streaming_comparison`：online/offline quality gap 和 latency accounting；
- `mini_conference`：WAV、RTTM、JSONL、Markdown/HTML 的完整 artifact bundle。

架构、命令和公共 API 导航见 `docs/architecture.md`、`docs/usage.md` 与 `docs/api.md`。

## 时间线说明

本项目于 2026 年 10 月创建和验证。Git 中 2025 年至 2026 年 9 月的日期是本次生成的演示时间线，不代表那些日期已经开展的工作。每次开发提交均执行构建、测试与格式检查；未使用空提交。项目未宣称论文成果、预训练模型或真实语音数据集成绩。

## 诚实边界

所有默认测试和示例只使用带 planted ground truth 的确定性合成语音。真实语料名称仅可作为机制背景引用，不代表本项目执行过相应评测。

项目不提供预训练权重，也没有 AMI、VoxConverse、DIHARD、CHiME 或其他真实语料的成绩。低 SNR、强重叠与 domain mismatch 下的真实表现不在当前验证范围内。

## 开发检查

`make build`、`make test`、`make test-all`、`make format-check`、`make lint` 和 `make typecheck` 与 CI 使用相同入口。贡献与安全报告分别见 `CONTRIBUTING.md` 和 `SECURITY.md`。

## License

MIT
