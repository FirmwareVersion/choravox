# 贡献指南

感谢改进 choravox。提交应聚焦一个可解释的行为变化，并同时提供边界测试或序列化 golden check。请勿加入真实录音、外部语料下载、预训练权重、GPU 必需路径或无法复现的指标。

## 本地验证

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
make format
make build
make test-all
make format-check
make lint
make typecheck
```

需要模型测试时安装 `.[dev,torch]`，并使用 PyTorch CPU wheel index。训练或较慢集成测试必须标记 `slow`；PyTorch 测试还必须标记 `model`，且在 extra 未安装时 cleanly skip。

## 变更约束

- 所有随机过程接收显式 seed，同配置应产生 byte-identical artifact。
- 新 schema 必须有版本、严格解析、完整性校验和 roundtrip/golden 测试。
- 合成实验的数字必须由测试或示例真实计算，并明确标注 synthetic/planted/oracle adapter。
- 公共 API 使用 dataclass 与完整 type hints；提交前运行全部 Makefile 检查。

Pull request 请说明动机、验证命令、兼容性影响和 honesty boundary。
