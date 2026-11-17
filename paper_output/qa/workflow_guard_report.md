# Workflow Guard Report

- Target step: `S1`
- Status: `FAIL`
- Generated at: `2026-11-17T15:23:16`
- Skill: `authoritative-data-harvester`
- Required step: `S1`
- Handoff: 外部数据检索必须围绕已解析的题意和子问题进行，检索结果回填 crawled_data/ 或 data_plan。
- Next action: 禁止启动 authoritative-data-harvester；先补齐 S1 及之前的失败项。

## Steps
- S0 准入预检: `FAIL`
  - 缺少文件：paper_output/preflight_report.json
  - 缺少文件：paper_output/input_manifest.json
- S1 审题分析: `FAIL`
  - 缺少文件：paper_output/step1/problem_analysis.json
