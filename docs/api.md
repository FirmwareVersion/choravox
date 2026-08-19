# 公共 API 导航

稳定数据记录位于 `choravox.types` 和 `choravox.config`。常用入口如下：

- `MeetingFactory.build()`：生成 waveform、dialogue script 和 planted word timestamps。
- `detect_speech()`：返回 frame energy、adaptive threshold、posterior 与 speech decisions。
- `embed_features()`、`diarize_embeddings()`：生成 statistics embedding 并执行聚类、VB refinement 和 HMM smoothing。
- `decode_speaker_attributed()`、`forced_align()`：执行 SOT attribution 与 CTC Viterbi alignment。
- `cpwer()`、`diarization_error()`、`jaccard_error_rate()`：计算 permutation-invariant transcript 和 speaker metrics。
- `streaming_latency()`、`compare_online_offline()`：分解 buffering/algorithmic latency 并比较同一合成会议。
- `synthesize_report()`、`write_report_artifacts()`：生成并校验 Markdown/HTML report bundle。

RTTM、STM、JSONL、embedding、manifest、evaluation 和 report manifest 都带 schema version 与完整性摘要；解析器对未知版本和摘要不一致执行严格拒绝。
