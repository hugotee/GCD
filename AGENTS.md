# 项目协作约定

- 本仓库为独立的完整预训练 Grounding DINO-T + Single LoRA + CIOD 基线；先读 README.md 与 docs/experiments/BASELINE_STATUS.md。
- 最新本轮用户授权：仅物理 GPU2。运行前检查占用，不停止他人作业，不默认使用其他GPU。
- 不引入 DGS 动态分组、任务路由、专家库、参数合并；当前不拼装五篇参考论文的方法。
- 无回放VOC07 15+5协议见 docs/experiments/VOC15_5_PROTOCOL.md。正式配置900 query、固定多尺度与完整数据；工程缩小配置须明确标记。
- 历史结果在 evidence/reference；本轮验证在 evidence/current。不得将历史、smoke或CPU工程结果冒充新正式实验。
- checkpoint/optimizer/scheduler/loop状态必须一致恢复；禁止静默丢参数。失败日志保留，不覆盖旧实验目录。
- 不自动启动数十小时训练。用户已授权每完成一项任务后提交并推送GitHub；只做正常推送，不覆盖远端历史。凭据不写入文件、Git或日志。
- 用户文档字体：英文/数字 Times New Roman，中文宋体（SimSun）；生成排版文档时核实字体真实可用。Markdown本身不指定渲染字体，不将替代字体伪称指定字体。
