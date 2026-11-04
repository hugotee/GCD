# GPU2 完整路径实测与预算

测量日期：2026-11-04。用户本轮最新指令明确使用物理GPU2，实际为RTX3090 24GiB；没有使用其他GPU，没有停止他人进程。开始前GPU2为12MiB、0%利用率。此前GPU0繁忙的审计是旧授权阶段快照，不代表本轮在GPU0上执行。

## 本轮实际执行

- Task1和Task2各8微步smoke，Task2严格加载Task1工程checkpoint并继续同一个LoRA；两阶段均完成检测损失、反传、更新、冻结参数hash检查和4张训练侧图片的保存/重载预测完全一致验证。
- 两阶段随后各200微步profile，预热20微步；batch2/accum2，有效batch4，每阶段400次图像曝光、100次优化更新。稳态统计180微步/360图；不是完成200个完整epoch，也不是200张不重复图。
- 完整预训练Grounding DINO-T，24个FFN LoRA、r16/alpha32、688128可训练参数，总参数173600278。900query，训练短边640/672/704/736/768/800、长边不超过1333；没有降低GPU测试分辨率、query数或训练数据池。
- 每微步CUDA同步；在MMEngine logger重置peak统计前采样，记录allocated和reserved。稳态时间不含模型加载、保存、参数hash或评价。CPU集成另用85×128实际输入和20query，只作正确性检查，绝不计入此表。

## 新实现单卡GPU2测量

|阶段|微步 / 预热|秒/微步|images/s|峰值allocated GiB|峰值reserved GiB|有效更新 / AMP跳步|每epoch微步|12epoch纯训练估算h|
|---|---|---:|---:|---:|---:|---|---:|---:|
|task1|200 / 20|0.5255|3.806|6.88|13.98|100 / 0|2237|3.92|
|task2|200 / 20|0.5444|3.674|7.29|13.73|100 / 0|503|0.91|

两阶段12epoch纯训练合计约 **4.83小时**，单卡约同等GPU小时；未包含完整test推理、加载、checkpoint、hash和共享服务器负载变化，不作为完成时间承诺。各阶段profile审计区间墙钟约112.4/116.3秒，启动和前置模型加载另计。当前没有启动新正式B1/B2。

GPU利用率另用只读nvidia-smi每2秒采样，共147次，均值35.5%、最大100%。采样覆盖加载、保存和阶段之间的等待，因此这个均值不是纯稳态计算利用率，不能与稳态images/s混为一项。

原始证据：evidence/current/profile_task1、profile_task2中的audit_rank0.json和timings_rank0.json；smoke_task1、smoke_task2中的reload_check.json；gpu_utilization.json和profile_summary.json。两阶段冻结参数未变、48个LoRA张量发生更新、梯度有限且非零；全部200次优化更新有效，无AMP跳步。

## 历史成本，单独保存

旧GDINO-CIOD曾测量其他授权下的单卡和GPU1/2 DDP，记录在evidence/reference/evidence/profile_summary.json。其双卡12epoch训练估算2.35小时，正式历史流水线实耗约3.18小时；其中B1约2.21小时、B2约0.51小时，余量含评价等。它不是本轮新实现的双卡实测；当前仅GPU2获使用范围，不自动启动DDP。

两张24GiB卡不能作为共享48GiB使用。若将来加入teacher/KD、改变插入位置或扩充COCO数据，须重新测完整路径，不能沿用本表承诺显存和时间。COCO70+10/40+40尚无本轮实际预算。
