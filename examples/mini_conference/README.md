# Mini-conference 完整产物

本示例通过公开 CLI 生成合成会议 WAV、RTTM、versioned transcript、metadata，以及自包含 Markdown/HTML 报告。报告中的 VAD 帧数来自实际运行；diarization 与 transcript 使用 planted/oracle adapter，因此 DER 和 cpWER 用于验证管线与报告机制，不是未知音频上的模型精度。

```bash
.venv/bin/python examples/mini_conference/run.py build/example-conference
```

`summary.json` 记录 report digest 和完整文件清单，可用于复现检查。
