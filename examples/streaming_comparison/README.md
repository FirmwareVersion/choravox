# Online 与 offline 对比

本示例在同一个 planted dialogue 上比较 offline identity reference 与 block-quantized online segmentation，使用库内 DER、cpWER 和 latency accounting 生成三组实际测量点。最大 block 条件还模拟一次不稳定词修订，用于展示 cpWER gap；它不是生产流式识别器的实测结果。

```bash
.venv/bin/python examples/streaming_comparison/run.py build/example-streaming
```

输出的 `latency-quality.html` 内嵌两张 SVG；它和 `summary.json` 仅描述本次确定性合成实验。
