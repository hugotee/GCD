# 基线建设状态

更新：2026-11-05。本仓库根目录为 /Users/hugo/Documents/ChatGPT/GCD，origin=github.com/hugotee/GCD；远程最初为空。独立搭建，不把旧DGS项目作为主体。用户最新允许物理GPU0–3中选择空闲卡；已有训练/探针仍按原记录使用GPU2，all20完整评价已迁到GPU1。不启动新的正式长训练。

## 已实现

|部分|实际状态|证据/入口|
|---|---|---|
|文献|五篇原文/作者代码定向核查，未知项明示；3个可否定研究问题|docs/research/literature_matrix.md、research_questions.md|
|基础模型|完整OGC权重，SHA256检查；官方MMDetection固定提交|src/gcd/model.py、configs/model/gdino_t.py|
|Single LoRA|项目内实现；24个视觉/文本encoder FFN Linear，r16/alpha32，688128可训练参数|src/gcd/peft.py；没有运行时DGS导入|
|数据|从VOC原始XML重新生成；逐类positive文件独立核验，4473/1006图、468重现|scripts/prepare_voc.py、服务器data/voc15_5/manifest.json|
|训练/恢复|冻结模型、B1、B2、smoke、profile、严格模型重载；同阶段optimizer/scheduler恢复和配置指纹|scripts/run.py、src/gcd/hooks.py、runtime.py|
|评价|完整20类prompt；VOC07 AP50、COCO AP50/AP50:95、逐类和分组AP|scripts/evaluate.py、summarize_results.py|
|正确性|单元测试、完整网络CPU前反向/更新、冻结哈希、重载预测一致|evidence/current/unit_tests_final.log、cpu_integration.json|
|GPU工程路径|GPU2的完整多尺度/900query smoke与200step profile|实际检查结果和成本见GPU_PROFILE.md及evidence/current|
|真实基线结果|导入既有独立GDINO-CIOD已完成B0/B1/B2，核对原始预测与checkpoint SHA、逐类均值、B2初始化链|evidence/reference/import_manifest.json、evidence/current/asset_audit.json|
|B0评价复核|旧冻结模型4952图真实预测按文件名重映射ID，在新XML GT和新评价器上重算一致|evidence/current/b0_recomputed.json；不是新增GPU推理|

## 真实参考结果

来源是用户已有独立GDINO-CIOD在2026-11-04完成的正式运行，seed20261103、固定字母类序、B1/B2各12epoch、历史GPU1/2 DDP。当前仓库没有重新训练这些正式模型。历史元数据中的GPU编号只描述来源，不授权当前使用它们。

VOC2007 11点AP50（百分数）：

|Model|Old15|New5|All20|
|---|---:|---:|---:|
|B0 Frozen|77.0205|70.2854|75.3367|
|B1 Task1|90.4050|78.9303|87.5363|
|B2 Task2|71.2727|82.2121|74.0076|

COCO风格AP@[.50:.95]，仍评价VOC测试集：

|Model|Old15|New5|All20|
|---|---:|---:|---:|
|B0|66.0300|59.7401|64.4575|
|B1|81.5731|68.9695|78.4222|
|B2|61.8329|72.2117|64.4276|

COCO AP50（不是上面的VOC07 AP50）：

|Model|Old15|New5|All20|
|---|---:|---:|---:|
|B0|81.0130|72.4785|78.8794|
|B1|95.7627|81.8120|92.2751|
|B2|73.6587|86.1348|76.7777|

VOC07 AP50的旧类遗忘19.1322点，新类适应收益3.2818点；COCO AP对应19.7402、3.2423点。B2相对B0：old−5.7478、新类+11.9267、all−1.3292个VOC07 AP50点；all COCO AP仅−0.0299点。全类均值接近B0掩盖了明显的旧类损失与新类收益，不能据此声称无遗忘。

逐类变化已重新计算到evidence/current/per_class_changes.csv。旧类下降最大：person74.1651、chair54.3922、diningtable42.4109、bottle24.6860、motorbike20.3850点VOC07 AP50。这里只确定现象，不从该排序断定文本漂移或参数冲突为原因。

## 审计发现与已处理问题

1. 本地Mac无PyTorch/CUDA，不能声称本机具备3090训练能力；实际验证在已连接服务器进行。
2. 旧项目运行时从DGS加载LoRA类且mmdet为旧editable源。本轮LoRA移入项目，官方v3.3.0独立源码优先加载并校验关键文件hash；历史模型/损失对应文件已做差异审阅。
3. 官方DINO的dn_cfg=None构造与dn_meta=None分割存在接口问题。本项目用两个最小兼容接口关闭DN，保留原生Hungarian、Focal、L1、GIoU和decoder/encoder辅助损失；不是复制DGS核心机制。
4. MMCV Linear为nn.Linear子类，注入器必须用isinstance而非严格类型相等；回归测试覆盖该情况。
5. 不同转换器的数字image_id不同，直接复用历史预测会错误。当前按文件名建立映射，测试GT语义一致，B0重新评价完全一致。
6. COCO evaluator返回numpy整数，落盘前转为Python int；完美20类检测fixture同时核对三套AP=100及JSON序列化。
7. 早期失败尝试日志保留在evidence/current，不将失败或缩小CPU测试当成正式结果。原生Swin“无预训练”初始化日志出现在显式W0加载前，以严格state加载与W0 hash判断实际初始化。

## 尚未完成与后续

- 没有在新仓库代码下重跑完整正式B1/B2；新GPU工程测试与历史结果分开保存，不冒充方法复现。
- 同阶段完整MMEngine长训练中断恢复尚未进行长程验证；严格state与optimizer下一步数值恢复已由测试覆盖。
- 用户已确认COCO2017尚未准备；70+10/40+40尚未冻结、未启动训练。已核对官方类序/原始ID与提示token长度，见[COCO准备核查](COCO_DATA_READINESS.md)。
- 未实现五篇完整方法、基础KD或复杂创新模块。
- 旧类共现/候选分析、固定checkpoint提示/文本探针和new5/all20训练提示短程对照均已完成。下一步优先固定all20前向，仅控制旧token分类损失并检查损失归一化；长程/多seed复核另立计划，先复核GPU_PROFILE预算，不重复提交已完成流水线。

## 首轮机制诊断更新（2026-11-04）

已完成完整缓存预测分析、类别一致的共现分层、1052个隐藏旧框暴露统计，以及GPU2上128图/900query的六条件配对前向。报告见[首轮机制诊断](MECHANISM_DIAGNOSTICS_V1.md)。21项测试通过；新增训练0次，当前无排队作业。初步证据指向候选存在时旧类打分功能退化、encoder后文本变化；不将其直接认定为唯一因果机制或创新点。

## 固定1epoch训练提示对照完成（2026-11-05）

两组均从同一B1开始，503微步/252更新，1006图像及增广后输入/全局GT逐批完全一致。两个最终模型都完成4952图、all20提示评价。主要VOC07 AP50：

|模型|旧15类|新5类|全部20类|
|---|---:|---:|---:|
|B1起点|90.4050|78.9303|87.5363|
|new5训练提示|90.4654|86.7800|89.5440|
|all20训练提示|85.1257|84.7089|85.0215|

28项测试通过，完整初始化/状态校验通过。此前GPU迁移入口的初始化遗漏已修复，无效近零AP和中断尝试明确排除。报告见[TASK2_PROMPT_CONTROL_V1.md](TASK2_PROMPT_CONTROL_V1.md)。这是单seed短程筛查，不替代12epoch基线，也不是新方法结论。
