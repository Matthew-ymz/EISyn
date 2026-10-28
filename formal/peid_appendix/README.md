# PEID 附录的第一批 Lean 核验

本目录验证 2026-10-27 实时导出的 Overleaf 附录中，层次可加性、信息分解的代数恒等式及两源 PID 的条件推理。完整说明见 [核验报告](../../docs/reports/appendix_lean_verification_20261027.md)。

只依赖已经安装的 Lean 4.29.0；不需要 Mathlib，不修改系统默认版本，不下载新工具链。`lean-toolchain` 固定本目录的版本。

```sh
python formal/peid_appendix/verify.py
```

验证程序编译 `Appendix.lean` 和 `Counterexamples.lean`，检查全部命名定理的公理依赖，并保存 `verification.log` 与 `verification.json`。不允许未完成证明或自定义公理；正常结果只含 Lean 的标准逻辑公理。后一个文件给出 S4.2 关于特征向量变号的精确反例。

重要边界：信息量是抽象有序环/域中的数值；涉及概率论的结论以明确列出的熵恒等式、独立性后果、数据处理和非负性为前提。这里没有把 Shannon 熵、KL 散度或条件分布从底层定义重新实现。条件推理的通过不能写成“整份附录全部形式化验证完成”。

`source/provenance.json` 记录源文件校验值、在线项目和 Zotero 核对记录。`source/appendix_S1_S4_20261027.tex` 保留此次核验使用的 S1–S4 源码摘录，删去整行注释、保留其余原文，未修改原在线文件。`source/appendix_compiled_20261027.txt` 来自同次下载的附录 PDF，用于确认当前公式编号。
