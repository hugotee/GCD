# VOC2007 15+5 固定协议 v1

冻结日期：2026-11-04。任务为CIOD，不是VOC→Clipart跨域；初始完整预训练Grounding DINO-T，Single LoRA，无旧图像回放buffer、无蒸馏。

## 来源与类别顺序

使用标准Pascal VOC2007 trainval 5011图、test 4952图。类别顺序和当前类positive图像并集参考 [MMA作者公开VOC加载器](https://github.com/fcdl94/MMA/blob/65124ec1e2c8c96ff9886784f4e340eac0b22df8/maskrcnn_benchmark/data/datasets/voc.py)。这是明确可执行来源，不能声称所有VOC增量论文均用完全相同选图规则；GCD官方主要提供COCO配置。

|阶段|原类别ID（1起）|模型label（0起）|类别|
|---|---|---|---|
|Task1|1–15|0–14|aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person|
|Task2|16–20|15–19|pottedplant, sheep, sofa, train, tvmonitor|

不重命名或擅自替换同义词；测试固定上述20类、固定顺序。输入sample.text为原生MMDetection类别列表，由其生成类别文本与positive map。Task1训练只提供旧15类prompt；Task2训练提供累计20类prompt、GT只含新5类，label保持15–19。

## 选图与标注

- 每阶段取至少包含一个当前类非difficult对象的trainval图像，只保留当前类非difficult GT；用原ImageSets/Main/*_trainval.txt中的positive标记独立核验图像集合。
- Task1：4473图/10968框；Task2：1006图/1640框；468图自然跨阶段重现。不抽取旧类专用图、不建旧样本buffer，因此是current-positive exemplar-free设置；不是严格图像互斥。
- Task2图中旧类标注不可见，原生分类损失可能将旧对象当背景。这是无保持基线的已知冲突，不使用隐藏旧GT做oracle ignore。
- 原始XML/JPEG/split均只读；scripts/prepare_voc.py生成新COCO JSON。测试保留difficult并按指标ignore；训练排除difficult。
- XML的1起inclusive框转换为COCO [xmin−1,ymin−1,xmax−xmin+1,ymax−ymin+1]。新JSON图像ID采用原文件名数值；历史转换使用顺序ID，重用预测必须先按文件名映射，不能直接套用整数ID。
- 新生成数据与历史数据逐文件名、类别、框和difficult状态对齐；序列化hash不同不代表协议改变。完整manifest记录实际路径、阶段hash和testhash。禁止把旧DGS划分直接作为来源。

## 模型与优化

W0：官方MMDetection完整Grounding DINO-T OGC，SHA256 822d7e9db9ce6ff2119b72dc6e78606a1b0a2c307234798adf0cab50f1b424e3。上游检测预训练为O365/GoldG/Cap4M；VOC新增5类不是“预训练从未见过”的概念。

只在视觉encoder FFN与文本encoder FFN共24个Linear插入同一个逻辑adapter（各层A/B，非多专家），r16/alpha32/dropout0；688128个可训练参数。其余172912150个参数冻结；backbone、BERT保持eval。没有检测头、fusion层、neck、decoder额外训练参数；只有LoRA更新。

两阶段各12epoch，AdamW lr1e−4/wd1e−4，第10epoch后lr×0.1，clip0.1，AMP初始scale128，seed20261103。单卡batch2、accum2，有效batch4。训练短边随机640/672/704/736/768/800，长边≤1333；测试短边800、长边≤1333；900query、max_per_image300。不偷偷下采样数据或改类别顺序。关闭DN和activation checkpoint是显式工程选择，不能据此声称复现GCD训练。

## B0/B1/B2 与评价

B0直接加载W0，冻结评价；B1从W0学习旧15类；B2从B1末尾checkpoint继续同一LoRA，仅学习新类，不回放、不KD。B2跨阶段重置optimizer/scheduler；同阶段中断恢复必须保留完整状态。所有状态在同一个4952图test、全部20类提示下评价；不输入oracle Task ID，不按任务挑模型。

主指标为VOC2007 11点AP50，IoU匹配严格>0.5，difficult忽略、重复检测记FP；另列COCO AP50和AP@[.50:.95]（101 recall、maxDets100）。COCO用iscrowd近似保留difficult，不宣称两套ignore语义完全相同；VOC07主指标使用原difficult标志。逐类AP先算，再分别等权平均15/5/20类。预测不额外加未记录阈值或NMS。

F_old=OldAP(B1)−OldAP(B2)，G_new=NewAP(B2)−NewAP(B1)，各指标分别计算；同时报告B1/B2相对B0差异。B1的新5类是增量未监督能力；B2后VOC20类都已适配，不能用它们证明额外未适配类别保留。

trainval仅训练，test不用于选择epoch、调参或筛选改进模块；评价固定末尾12epoch。当前无独立调参集，若要搜索超参数，应在训练侧预留验证并形成新协议。单seed结果不构成稳定方法结论。
