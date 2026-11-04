# Task2训练提示单变量对照：预注册

用户授权继续研究、仅使用GPU2，每项任务完成后提交并推送GitHub。已完成的基线与机制诊断提交433c2ce、b5bafce已上传main。本对照在hugo/task2-prompt-control分支保留过程。

## 问题与预算

首轮诊断发现：旧类高IoU候选的打分下降，移除评测新增类提示未明显恢复，Task2存在1052个隐藏旧框。先检验训练提示是否影响早期旧类保持，不引入新方法。

固定同一真实B1（SHA256 af92d30f638a4d74634185c50129320257d1616462bb87966dba27ea0d0f471e）、seed20261103、Task2全部1006图/1640个新类框、FFN LoRA r16/alpha32、原生损失、900query、原多尺度、batch2/累积2、AdamW lr1e-4。两组都完整训练1epoch、503微步、预计252次更新；最后不足一个累积批次按原生MMEngine处理。学习率全程固定，优化器均从头建立，未沿用B1动量。

- new5：文本仅pottedplant/sheep/sofa/train/tvmonitor，数据集局部label为0–4。
- all20：文本为完整20类，同一新类GT的局部label为15–19。

每批记录图像ID、增广后像素、GT框和还原到全局空间的类别hash。两组必须完全一致；不把局部标签重编号当成监督差别。首批还核验原生正token映射实际对应正确类别。记录初始LoRA hash、冻结参数hash、有效更新和AMP跳步。

两组纯训练预计约9–10分钟，模型加载、checkpoint、hash与完整test评价另计。该预算来自GPU2已测503微步/epoch，不是完成时间承诺。先各8微步预检，再执行固定1epoch；预检checkpoint不进入正式对照。

## 预定评价与解释范围

固定最后一个epoch，不按test结果选择checkpoint或seed。两个模型统一all20 prompt，在完整4952图VOC07 test评价VOC07 AP50、COCO AP50/AP50:95及逐类15/5/20类平均；与相同B1比较保持和新类变化。先检验输入配对和优化有效性，通过后才比较结果。

这是一轮单seed的1epoch机制筛查，不替代12epoch基线。若new5保持旧类更好，还须检查新类适应是否同步减弱。new5同时改变文本条件、query选择及匹配，不能把差异全部解释为纯粹移除负损失通道。若没有明显改善，则训练提示负监督不足以独立解释遗忘。无隐藏旧标注进入优化，无旧样本回放，无KD。

## 运行与材料

使用scripts/prepare_prompt_control.py生成不可覆盖的plan.json，固定数据、权重、源码与预算hash；scripts/train_prompt_control.py复用未修改的baseline dry-run配置，仅覆盖已注册的提示/标签和训练预算。配置模板与实际配置分开保留。所有输出使用新目录，失败日志不覆盖。

计划与最终结果放在evidence/prompt_control/v1；模型和大型预测留在服务器outputs。完成后在本报告追加真实结果，不用预期代替实测。

## 预检结果

23项单元测试通过。new5/all20均完成完整分辨率、900query的8微步GPU2预检，各4次有效更新、无AMP跳步，48个LoRA张量更新、冻结参数不变。两组8批的图像ID、增广像素和全局GT指纹完全一致，初始LoRA hash相同；原生正token监督实际指向相同新类别。预检模型不进入后续对照，两个1epoch模型都重新从相同原始B1初始化。证据见evidence/prompt_control/v1/preflight_pair.json及preflight目录。
