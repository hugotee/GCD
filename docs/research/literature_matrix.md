# 五篇核心文献对比

核验日期：2026-11-04。证据为论文正文/补充、作者仓库的固定提交。这里不复现五种完整方法；论文间的原表不构成同一 benchmark。未核实项保留未知，作者局限与本项目判断分开写。

## 身份与任务

|简称|完整标题、作者|年份/状态|原问题与现有方法不足|任务和本项目关联|
|---|---|---|---|---|
|GCD|**GCD: Advancing Vision-Language Models for Incremental Object Detection via Global Alignment and Correspondence Distillation**；Xu Wang, Zilei Wang, Zihan Lin|AAAI 2025|局部阶段对齐破坏语义结构；文本参与编码/选 query/解码，使普通师生蒸馏对应失配|COCO 类增量 CIOD；直接机制参考，但其预训练条件不同于本项目|
|C²Path|**C²Path: Class-Conditional Pathway Decoupling for Vision-Language Incremental Object Detection**；Lecheng Xu, Feifei Shao, Ouyangzi Ye, Zhen Wang, Lin Li, Kexin Li, Zhao Wang, Changqin Huang|arXiv:2608.21937v1，2026-08-22；未据预印本认定正式接收|共享更新侵蚀类别边界，多类编码造成表征纠缠|COCO 两阶段 CIOD；直接类别干扰参考|
|DitHub|**DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection**；Chiara Cappellino, Gianluca Mancusi, Matteo Mosconi, Angelo Porrello, Simone Calderara, Rita Cucchiara|NeurIPS 2025|单体权重难以选择性更新类别；重复类别跨域重现时需要复用和组合知识|IVLOD，异质域/类别变化且允许类重现；属于本项目相邻设置|
|ZiRa|**Zero-shot Generalizable Incremental Learning for Vision-Language Object Detection**；Jieren Deng, Haojian Zhang, Kun Ding, Jianhua Hu, Xingxuan Zhang, Yunkuan Wang|NeurIPS 2024|闭集增量方法损害预训练零样本能力；旧任务与预训练知识均会遗忘|IVLOD；直接支持预训练能力保持问题，不是 VOC15+5 结果|
|MR-GDINO|**MR-GDINO: Efficient Open-World Continual Object Detection**；Bowen Dong, Zitong Huang, Guanglei Yang, Lei Zhang, Wangmeng Zuo|arXiv:2412.15979v2，2024；本次未核实正式会议归属|已有持续适配方法遗忘已见/未见类别；任务局部提示和 COCO 评价覆盖有限|OW-COD：连续少样本适配与开放词汇能力保持；相邻设置|

CIOD 指类别集合逐步增加；DIOD 指主要类别语义保持而领域改变；CDIOD/DuIOD 指类与域同时变化，具体交叠和测试组合须依论文定义。IVLOD 是视觉语言检测的增量适配目标，不能自动等同纯 CIOD。这里 MR-GDINO 的 open-world 主要评测文本指定未适配类的检测能力，不能等同“先发现 unknown 再标注”的 OWOD 协议。

## 模型、训练与可用信息

|维度|GCD|C²Path|DitHub|ZiRa|MR-GDINO|
|---|---|---|---|---|---|
|基础检测器|Grounding DINO-T|Grounding DINO Swin-T|Grounding DINO-T|Grounding DINO-T；另验证 OV-DINO|预训练 Grounding DINO Swin-T|
|完整检测器预训练|主文明确不用 O365 检测预训练；ImageNet/BERT 初始化不等于无任何预训练|说明沿用 GCD；未明确核实完整 OGC 权重加载，不能直接标记与本项目 W0 相同|使用官方 OGC 预训练权重|使用 O365/GoldG/Cap4M 预训练权重|使用冻结预训练模型；另有 COCO-ft 初始化分析|
|适配形式|常规模型更新+GA/CRD/CTD，不是 LoRA 方法|类别 token 条件 ClassLoRA+共享低秩基库+路由|encoder 类别 LoRA：类专属 A、共享 B|视觉卷积与语言线性投影旁的 RDB；不是标准 Single LoRA|文本 prompt 概念记忆+融合层 Q/K/V LoRA 交互记忆|
|参与训练参数|基阶段检测器；增量阶段语言骨干冻结，其他未冻结部分更新|基阶段不加模块；增量更新 neck/head、CEL/CCDM、文本自注意力；视觉 backbone/自注意力以 0.1× LR 微调；文本 backbone、decoder 冻结|冻结基础模型；优化 warmup、当前类别 A 和共享 B|冻结原模型；仅双侧高/低学习率适配分支及缩放等适配参数|冻结原模型；更新当前 prompt 与交互 LoRA，可分两阶段训练|
|旧样本回放/存储|无旧图像 exemplar buffer；保留教师，当前图可能含旧对象|声明 exemplar-free；沿用师生蒸馏|无旧图像回放；保存类别参数库|无旧样本/完整教师副本；保留原始权重与累计分支|无旧样本回放；保存参数记忆和视觉检索键，非零历史状态|
|Task ID|训练知道当前/旧类别边界；测试累计类别 prompt，不按 oracle Task ID 选模型|按类别 token 路由，未要求数值 Task ID；训练阶段边界已知|不必输入数值 Task ID，但原 IVLOD 评价使用任务局部类别提示，含任务语义先验|训练知道任务边界以合并/重置分支；原评价按数据集局部类别提示，不应称完全任务不可知统一评价|图像检索代替 oracle Task ID；oracle retrieval 只作分析|
|旧知识保持|旧类伪标签、对应响应和同模态原型拓扑|在 GCD 蒸馏上增加 token 条件软解耦；共享基与路由不构成数学上的硬隔离|历史类 A 保留、重复类初始化合并、共享 B 平滑更新|限制整个 RDB 输出以保留预训练能力，限制高 LR 分支以保留旧任务|检索旧记忆；无匹配时退回冻结模型|
|新类别适应|新类 GT 与旧伪标签累计对齐|当前类检测损失及动态 ClassLoRA 更新|任务 warmup 后类别专门化|高 LR 分支快速学习，任务结束并入低 LR 分支|当前概念 prompt/跨模态 LoRA 监督学习|
|开放词汇/零样本|架构为 VLD；主实验未独立评测未适配类别保留|主实验未独立评测零样本保留|显式报告 ZCOCO；未入专家库类依靠基础模型|显式目标与 ZCOCO 指标|显式 old/new/unseen，LVIS 评价；unseen 指增量未适配，不证明预训练绝未见|

参数范围依据 GCD §Implementation、C²Path §4.1，以及各官方脚本/配置；未把通用配置中的可选 Adapter/MoE 分支当成论文主方法。

## 实验、消融和边界

|论文|数据、设置、指标|核心模块与可核对消融|作者局限及本项目边界|
|---|---|---|---|
|GCD|COCO2017：40+40、70+10、40+10×4、40+20×2；COCO AP、old AP、FPP|70+10：Global+Pseudo 45.3 AP；直接 RD 42.9；CRD 46.0；再加 CTD 46.7（Table 2）|补充指出无旧对象共现时监督不足，新类与跨域仍有困难。本项目：不能把伪标签全部收益归给蒸馏；CTD 分别约束对象—对象、文本—文本距离，不直接固定跨模态相对位置|
|C²Path|COCO2017 40+40、70+10；AP/AP50/AP75/尺度 AP|70+10：GCD 46.7→均匀 CEL 48.4→CCDM 48.7（Table 2）；类 token 合成后只注入融合层语言侧投影|未单列充分的局限章节。本项目判断：证据仅两阶段；路由净增益为0.3，不能将全部2.0收益归于类别路由；共享低秩基仍需检查干扰|
|DitHub|ODinW-13、ODinW-O 重现类；full 与1/5/10-shot；Avg、ZCOCO、遗忘/存储|Table 1 full：Avg62.19/ZCOCO47.01；Figure 7 warmup/合并消融；Appendix A 共享/独立 B 与插入位置|Appendix F 指合并系数经验性、缺乏显式域知识。本项目判断：共享 B 更新仍会改变旧 BA；多类提示的参数平均不保证无冲突；保留库不等于零遗忘|
|ZiRa|ODinW-13 顺序适配、COCO 零样本；Avg、ZCOCO、hAP；三 seed|Table 3：无 Rep+/ZiL 为39.72 ZCOCO/58.12 Avg；完整46.06/59.73；仅 ZiL 时 Avg54.24|§4.3 明示仅约束残差可能损害可塑性。本项目判断：小残差是代理量，不能凭随机噪声鲁棒性保证任意旧类功能不变|
|MR-GDINO|ODinW-13 的1/3/5/10-shot 连续学习，LVIS minival1203类；old/new/seen/unseen AP、平均 rank|10-shot Table 4：概念+交互无检索 old30.9/unseen10.7；加检索51.3/20.7；Table 7 有 oracle 检索对照|§5.4.4 提出大规模更精确检索值得研究。本项目判断：整图检索与多类场景匹配、记忆增长及阈值泛化需单独验证；不能将0.1%激活参数理解为总记忆常数|

上述论文数值仅解释其自身消融，不与本项目 VOC AP 比大小。DitHub 对 ZiRa 重新评测的 full Avg57.98，与 ZiRa 原文59.73不同，禁止混用为同一运行。

## 原文、代码与可复用性

|论文|论文依据|官方代码及本次检查|本项目处理|
|---|---|---|---|
|GCD|[AAAI 正式页](https://ojs.aaai.org/index.php/AAAI/article/view/32864)、主文§3–4/Table2及官方补充|[Never-wx/GCD](https://github.com/Never-wx/GCD/tree/420708a7cca19d320b082a859cc72039dbdd0dbf)，检查70+10增量配置、CRD/CTD参数、900 query；Apache-2.0|可参考 query 对应实现；当前不移植完整蒸馏|
|C²Path|[arXiv v1](https://arxiv.org/html/2608.21937v1)，§3.2、§4.1/Table2|论文页及题名/code检索未定位作者官方代码；这是检索状态，不断言永不发布|参数细节只记原文；不编造代码级实现|
|DitHub|[NeurIPS 正式页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/211bc037aefd17aec8e3e48a47cfa007-Abstract-Conference.html)、[全文v4](https://arxiv.org/html/2503.09271v4)|[chiara-cap/DitHub](https://github.com/chiara-cap/dithub/tree/16f37b42965f7da804bfea8a9570d6706444bdab)，检查 README、lora_utils.py、模块库目录；aimagelab/DitHub 的 README 指向此仓库|可参考参数冻结/状态管理，当前不引入专家组合；整体许可需按实际复用文件逐项核验，本次未复制其代码|
|ZiRa|[NeurIPS 正式页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f6f4b34d255c2c6c2391af975bed0428-Abstract-Conference.html)、[全文v3](https://arxiv.org/html/2403.01680v3)|[JarintotionDin/ZiRaGroundingDINO](https://github.com/JarintotionDin/ZiRaGroundingDINO/tree/10f0c90b444de23ebc880d4f3acc313b8952181c)，检查适配模块及train_odinw13_adapter.sh；Apache-2.0|参考分支合并、预训练能力评价，当前不实现 RDB/ZiL|
|MR-GDINO|[全文v2](https://arxiv.org/html/2412.15979v2)，§3–5|[DongSky/MR-GDINO](https://github.com/DongSky/MR-GDINO/tree/254095424cccdddf9a5f54b051ef4122acd73655)，检查 main.py、coop_lora配置及 README；README注明部分 OW-COD 配置尚待提供，不能称一键完整复现|参考记忆/检索机制，不导入其路由；未复制代码，不将未审计许可材料纳入项目|

完整网络与检测损失采用 [官方 MMDetection v3.3.0](https://github.com/open-mmlab/mmdetection/tree/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a)。没有复制五篇方法的核心实现。对原仓库的检查为静态核对，不代表成功运行或重现。
