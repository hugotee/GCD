# VOC与COCO基础框架接入

2026-11-05。本轮完善Grounding DINO完整预训练模型和Single LoRA的基础训练、评价流程。当前不实现论文创新模块，也不把训练路径检查当作正式增量结果。

执行范围：

1. 用原始预训练权重重新完成VOC07 test的4952图B0评价，与已有结果交叉核对。已有VOC框架可运行，本次补充新工程自己的完整推理证据。
2. 下载COCO2017的完整train2017、val2017与检测标注；核对公开MD5、ZIP CRC、全部文件、80类原始ID及数据数量。
3. 生成70+10、40+40阶段标注和可追溯manifest，接入原生训练/评价入口；框与crowd语义遵循COCO，不沿用VOC的difficult过滤。
4. 完成COCO完整5000图冻结模型评价、LoRA训练路径smoke与200微步性能测量，再报告正式训练预算。

只使用用户授权的GPU0–3中的空闲卡，启动前重新检查。下载与CPU数据处理不占GPU。各任务以独立终端会话运行，状态和日志留在outputs；完成一项后提交并推送GitHub。数据、权重和私有runtime配置不进入Git。

下载文件链接来自[COCO官方下载页](https://cocodataset.org/#download)，官方域名映射到同名S3桶；可使用有效TLS的路径地址。公开MD5同时核对[PaddleDetection数据清单](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/ppdet/utils/download.py)。原始归档保留，检测标注仅提取instances_train2017.json和instances_val2017.json；其余图像不抽样。

## 新完成的VOC冻结评价

物理GPU1重新完成4952图推理与评价，耗时约21.48分钟（含模型准备和两次指标汇总）。完整20类提示、900 query，直接使用原始预训练模型，不训练LoRA。

|指标|旧15类|新5类|全部20类|
|---|---:|---:|---:|
|VOC07 AP50|77.0205|70.2854|75.3367|
|COCO AP50:95，评价VOC测试集|66.0300|59.7401|64.4575|

与此前保存的B0复核数值一致。该结果确认新工程的完整推理与评价能够复现原始基础输出；不说明增量训练已完成，也不证明抗遗忘方法有效。原始日志、配置、来源快照和逐类结果见evidence/foundation/v1/voc_B0。

COCO接入代码已通过37项单元测试；全80类评价开始前，模型实际900个张量与W0逐项完全相等。COCO AP和吞吐尚待任务完成后填写。
