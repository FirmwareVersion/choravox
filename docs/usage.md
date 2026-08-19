# 使用指南

## 安装与检查

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
make build
make test
```

`make test` 跳过标记为 `slow` 的训练或集成测试；`make test-all` 包含全部 slow/model 测试。默认示例不要求 PyTorch。

## CLI

```bash
.venv/bin/choravox synthesize build/meeting --seed 7
.venv/bin/choravox vad build/meeting/meeting.wav build/vad.json
.venv/bin/choravox demo build/demo --seed 7
.venv/bin/choravox stream-demo build/stream --seed 7
```

其余子命令 `embed`、`diarize`、`overlap`、`transcribe`、`align`、`evaluate`、`compare`、`report` 使用 `.npy` 或 JSON 作为明确的中间产物。运行 `choravox COMMAND --help` 可查看参数。

## 示例

四个目录分别覆盖 oracle-embedding diarization、speaker-attributed CTC overlap sweep、online/offline latency-quality 对比和 mini-conference 报告。每个目录都有独立 README，输出均标记为 synthetic。
