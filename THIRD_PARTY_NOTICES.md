# 来源与复用记录

- 官方 [MMDetection](https://github.com/open-mmlab/mmdetection/tree/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a)，Apache-2.0。通过固定提交独立下载，源码及其LICENSE保留在external/mmdetection；许可证副本随本仓库保存在third_party/mmdetection.LICENSE；不将整个第三方仓库复制进本项目Git历史。
- Grounding DINO完整OGC权重与架构来自 IDEA-Research / OpenMMLab；来源与SHA256见模型审计和runtime模板。BERT原始文件单独指向本地预训练资源，不进Git。
- configs/model/gdino_t.py 的结构来自用户已有 GDINO-CIOD 的已解析原生模型配置；其上游为Apache-2.0 MMDetection。src/gcd、scripts、tests为本项目实现/重构，未复制五篇参考方法的核心代码。
- 历史结果只读导入自用户独立 GDINO-CIOD，commit 1bc53ea5fe61ee935ffaa42405ce36ea81b992f2。evidence/reference/import_manifest.json逐文件保存原始SHA256。旧目录和历史未改动。
- LoRALinear依据标准低秩残差公式实现，保持历史检查点的weight/bias/lora_A/lora_B键和数值运算顺序；运行时不读取DGS文件，也没有合并、分组、专家或任务路由。历史LoRA文件带Microsoft/MIT头，本次未将其复制进项目。
- 本项目两处DN兼容处理是对官方v3.3.0接口行为的显式适配：构造后用零query接口替换DN生成器；head在dn_meta为None时返回全部matching输出。检测分类、L1、GIoU和Hungarian匹配仍用上游实现。
- configs/datasets/coco80_reference.json的COCO类别事实与原始ID提取自[GCD作者固定版本METAINFO](https://github.com/Never-wx/GCD/blob/420708a7cca19d320b082a859cc72039dbdd0dbf/mmdet/datasets/coco_increment.py)，另附VOC语义对应；只参考数据协议，未引入其蒸馏或模型代码。

未为用户新代码擅自选择开源许可证。已按用户授权推送GitHub；上游MMDetection许可证副本随仓库保留，模型权重、BERT资源与数据集不进入Git。
