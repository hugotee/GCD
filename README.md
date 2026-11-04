# Grounding DINO + Single LoRA + CIOD

独立科研基线，连接至 https://github.com/hugotee/GCD 。本项目使用完整预训练 Grounding DINO-T，以同一个 LoRA 顺序学习 VOC2007 旧15类、新5类。后续 COCO70+10/40+40 尚未启动。没有动态分组、任务路由、专家库、教师蒸馏或参数合并。

## 当前可用材料

- [文献矩阵](docs/research/literature_matrix.md) 与 [候选问题](docs/research/research_questions.md)。
- [固定 VOC 协议](docs/experiments/VOC15_5_PROTOCOL.md)、[建设状态与结果](docs/experiments/BASELINE_STATUS.md)、[GPU 实测](docs/experiments/GPU_PROFILE.md)。
- src/gcd：原生模型包装、项目内 LoRA、XML划分、训练审计、AP评价。
- evidence/reference：此前 GDINO-CIOD 的真实B0/B1/B2结果与来源hash；不是本次重新训练。
- evidence/current：当前独立实现的测试、数据/资产复核、B0重算和GPU工程验证证据。

## 最新诊断

[首轮机制诊断](docs/experiments/MECHANISM_DIAGNOSTICS_V1.md)已完成缓存预测分析；已在GPU2完成128图完整raw900-query/文本特征配对探针，无新增训练或后台排队。诊断输入与状态见 evidence/diagnostics/v1。

## 环境

已在Linux服务器复用依赖完成验证；Mac本机没有CUDA/PyTorch，不作为训练环境。推荐新建Python3.10环境：

```bash
bash scripts/bootstrap_mmdet.sh
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
python -m pip install -r requirements.txt -r external/mmdetection/requirements/runtime.txt
python -m pip install -e .
```

模型、原生匹配和检测损失来自固定的官方 MMDetection v3.3.0，commit 44ebd17b145c2372c4b700bfb9cb20dbd28ab64a。启动器显式加载 external/mmdetection 并验证关键文件hash，避免导入环境中旧的DGS editable包。LoRA完全位于本项目；仅有两个关闭DN所需的兼容接口，不改动匹配与检测损失。

复制 configs/runtime.example.json 为 configs/runtime.local.json，填写完整权重、BERT、官方源码和数据manifest的绝对路径。local配置被Git忽略，不能写入密码。W0为[官方转换OGC权重](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth)，完整SHA256见配置；不是旧增量模型。

## 数据与正确性

```bash
python scripts/prepare_voc.py --voc-root /absolute/path/VOC2007 --out data/voc15_5
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/cpu_integration.py --runtime configs/runtime.local.json --out outputs/cpu_integration.json
```

CPU集成只用于正确性，显式改为小图/20query；不能作为GPU吞吐或正式AP。正式入口默认保持900query。

## GPU2 工程验证

以下使用物理GPU2，不再额外设置CUDA_VISIBLE_DEVICES或重编号。当前runtime只授权2；忙卡会拒绝启动。

```bash
python scripts/run.py smoke --runtime configs/runtime.local.json --gpus 2 --phase task1 --steps 8 --warmup 0 --verify --work-dir outputs/smoke_task1
python scripts/run.py profile --runtime configs/runtime.local.json --gpus 2 --phase task1 --steps 200 --warmup 20 --work-dir outputs/profile_task1
python scripts/run.py profile --runtime configs/runtime.local.json --gpus 2 --phase task2 --init outputs/profile_task1/iter_200.pth --steps 200 --warmup 20 --work-dir outputs/profile_task2
```

profile_task2从工程checkpoint继续同一个LoRA，只验证顺序路径，不是B2科研结果。目录已存在时拒绝覆盖；重试使用新目录保留失败证据。

## B0 / B1 / B2 正式入口

以下是准备好的命令，不由README自动执行。此前相同协议的完整结果已保留，不重复提交正式训练。

```bash
python scripts/run.py frozen --runtime configs/runtime.local.json --gpus 2 --work-dir outputs/B0_eval
python scripts/run.py train --runtime configs/runtime.local.json --gpus 2 --phase task1 --work-dir outputs/B1
python scripts/run.py eval --runtime configs/runtime.local.json --gpus 2 --phase task1 --init outputs/B1/epoch_12.pth --work-dir outputs/B1_eval
python scripts/run.py train --runtime configs/runtime.local.json --gpus 2 --phase task2 --init outputs/B1/epoch_12.pth --work-dir outputs/B2
python scripts/run.py eval --runtime configs/runtime.local.json --gpus 2 --phase task2 --init outputs/B2/epoch_12.pth --work-dir outputs/B2_eval
python scripts/evaluate.py --manifest data/voc15_5/manifest.json --predictions outputs/B0_eval/predictions.bbox.json --provenance outputs/B0_eval/provenance.json --model B0 --out outputs/results/B0
```

B1/B2评价分别替换上条命令的预测目录、model和out。evaluate.py校验完整4952图评价完成证据、类别顺序和标注hash，输出VOC07 11点AP50、COCO AP50/AP@[.50:.95]、逐类及15/5/20类平均。汇总：

```bash
python scripts/summarize_results.py --results-root outputs/results --out outputs/results.csv
```

同一阶段恢复用 --resume outputs/B1/epoch_N.pth 和原work-dir，保留optimizer/scheduler/loop状态；跨阶段用 --init，明确重置优化器。配置、模型模板、manifest和W0组成恢复指纹，改变后拒绝续训。--dry-run可只生成配置和来源记录。

参考实现与许可见 THIRD_PARTY_NOTICES.md。已按用户授权将基线与首轮诊断提交推送至GitHub main；后续任务在hugo/分支保留提交与结果。
