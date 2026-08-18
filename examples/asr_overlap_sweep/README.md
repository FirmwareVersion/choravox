# Speaker-attributed CTC 与重叠比例

本示例从确定性 dialogue script 取得 planted transcript，生成可解释的 CTC score matrix，执行 SOT speaker-change 解码和 speaker attribution，再测量不同 planted overlap ratio 下的 cpWER。为展示曲线，stress adapter 会按比例删除合成 token；它不是声学模型，也不是任何真实语料上的结果。

```bash
.venv/bin/python examples/asr_overlap_sweep/run.py build/example-asr
```

`summary.json` 记录每个条件的实际删除数和 cpWER，`overlap-cpwer.svg` 直接由这些测量值生成。
