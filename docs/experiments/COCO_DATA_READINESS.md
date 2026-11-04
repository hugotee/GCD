# COCO2017数据准备与协议核查

状态：用户已确认尚未准备COCO2017。当前只有VOC可运行数据，COCO70+10/40+40尚未冻结、未启动训练。已生成可核验的80类顺序/原始ID/VOC对应表，并用实际BERT tokenizer完成CPU文本长度检查。

## 已核实的参考协议

固定来源为GCD官方commit `420708a7cca19d320b082a859cc72039dbdd0dbf`，不使用本项目同名仓库作为方法来源：

- [官方两阶段切分脚本](https://github.com/Never-wx/GCD/blob/420708a7cca19d320b082a859cc72039dbdd0dbf/scripts/select_categories_2step.py)：按原始category ID排序，取前N类和其余类，保留当前类标注及至少含一条当前类标注的图像；默认N=40，70+10配置另列于作者配置目录。原始框与crowd字段保留，不能套用VOC排除difficult的规则。
- [官方增量数据集](https://github.com/Never-wx/GCD/blob/420708a7cca19d320b082a859cc72039dbdd0dbf/mmdet/datasets/coco_increment.py)：包含固定80类名称与不连续的原始ID，区分cur_text局部标签与full_text全局标签。
- [70+10增量配置](https://github.com/Never-wx/GCD/blob/420708a7cca19d320b082a859cc72039dbdd0dbf/configs/gdino_inc/70%2B10/gdino_inc_70%2B10_70-79_gcd_scratch_coco.py)及[40+40增量配置](https://github.com/Never-wx/GCD/blob/420708a7cca19d320b082a859cc72039dbdd0dbf/configs/gdino_inc/40%2B40/gdino_inc_40%2B40_40-79_gcd_scratch_coco.py)：Task2采用full_text，最终评价完整instances_val2017.json。两阶段可能复现相同图像，不能未经统计声称图像互斥；无回放应指没有另建旧样本缓冲。

70+10旧类最后是oven，新10类依次为toaster、sink、refrigerator、book、clock、vase、scissors、teddy bear、hair drier、toothbrush。40+40边界为bottle / wine glass。这里的0–79是顺序索引；例如toaster为索引70、原始ID80，不能用category_id−1当标签。

原官方基础模型配置仅指定Swin预训练且主体参与训练，不能把其报告AP直接当作本项目完整OGC+LoRA的同条件结果。本项目后续可以参考其类别协议，但必须另报预训练条件、LoRA范围、增广、提示和训练预算。

## 已完成的CPU验证

configs/datasets/coco80_reference.json从官方METAINFO解析生成，原始源码Git blob SHA1核对为`6f48ce390e22a1bd64404edd3f42c90744ca702a`，原始源码SHA256为`850ddd391eec6c25d0db2fdc5043b9d4df27e519a748bd1675bb7d9939882d52`。没有导入或执行作者的方法代码。

|提示类别集合|类别数|实际token数，含特殊token|低于256上限|
|---|---:|---:|---|
|完整COCO|80|195|是|
|前70类|70|170|是|
|前40类|40|101|是|
|扣除VOC20语义对应类|60|152|是|

VOC的aeroplane/motorbike/sofa/tvmonitor/diningtable/pottedplant分别对应COCO的airplane/motorcycle/couch/tv/dining table/potted plant。完整20类对应均已验证名称与ID唯一；剩余60类可作为后续VOC适配之外的评价候选，不能称为预训练从未见过的类别。上述CPU检查未初始化CUDA。

evidence/current/coco_readiness_20261104.json记录当前状态与hash。attempt01保留最初的文本提取快照；随后以官方Git blob核对原始字节，修正源码校验信息后重新生成最终快照，类别与token统计未变。

## 数据到位后的验收

需要完整train2017、val2017和annotations/instances_train2017.json、instances_val2017.json。先验证80类原始ID/名称、全部图像文件存在、训练/验证图像数量，再生成不可覆盖的阶段JSON与manifest，统计各阶段图像/实例数、交叠图像、隐藏旧框和crowd语义。训练图像筛选、无效框处理、min_size与原始ID映射均需留痕。正式profile需从当前已授权的GPU0–3选择空闲卡，不能沿用VOC数据规模估计COCO成本。

```bash
# 当前状态快照（已运行，输出已存在时拒绝覆盖）
CUDA_VISIBLE_DEVICES= python scripts/audit_coco_readiness.py \
  --runtime configs/runtime.local.json --user-confirmed-not-prepared \
  --out evidence/current/coco_readiness_20261104.json
# 数据准备后检查全部图像存在和80类映射；使用新的输出路径。
CUDA_VISIBLE_DEVICES= python scripts/audit_coco_readiness.py \
  --runtime configs/runtime.local.json --coco-root /absolute/path/coco2017 \
  --out outputs/coco_readiness_after_download.json
```
