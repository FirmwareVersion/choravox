## 变更

说明问题、实现边界和兼容性影响。

## 验证

- [ ] `make build`
- [ ] `make test-all`
- [ ] `make format-check`
- [ ] `make lint`
- [ ] `make typecheck`

## 可复现性与诚实边界

- [ ] 新随机路径使用显式 seed，序列化变化有 golden/roundtrip 覆盖。
- [ ] 所有指标来自实际执行并明确标注 synthetic；未暗示真实语料、预训练权重或论文结果。
- [ ] 默认路径不新增 GPU、外部数据集、付费服务或网络依赖。
- [ ] 如涉及 PyTorch，测试标记 `model` 并在 extra 缺失时 cleanly skip。
