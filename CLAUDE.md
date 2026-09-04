# InfoCom 代码解读与训练指南

本文档面向需要理解当前实现并复现 InfoCom 训练的开发者。默认命令从仓库根目录 `/data1/bohnsix/InfoCom` 执行。本文档只描述当前代码行为，不自动修复代码中的历史示例或潜在问题。

## 1. 项目定位与入口

InfoCom 是一个作用于 BEV 中间特征的通信压缩模块，核心由三部分组成：

1. 信息瓶颈编码：将每个 CAV 的 BEV 特征压缩为低维 latent 分布。
2. 稀疏空间 mask：选择需要传输的空间位置，并进行 4-bit 量化。
3. mask 引导的解码：根据 latent 和空间 mask 重建 BEV 特征，再交给协同融合网络。

关键入口：

- [README.md](README.md)：官方安装、数据和训练流程。
- [opencood/tools/train.py](opencood/tools/train.py)：训练不带通信模块的基础协同感知模型。
- [opencood/tools/train_infocom.py](opencood/tools/train_infocom.py)：加载基础 checkpoint，训练 InfoCom 通信模块。
- [opencood/tools/inference.py](opencood/tools/inference.py)：测试集推理和 AP 评估。
- [opencood/tools/train_utils.py](opencood/tools/train_utils.py)：动态创建模型和 loss，加载 checkpoint，设置 optimizer/scheduler。
- [opencood/extensions/coib_communication.py](opencood/extensions/coib_communication.py)：当前 multiscale InfoCom 使用的通信模块。
- [opencood/extensions/coib_encoder.py](opencood/extensions/coib_encoder.py)：CoIB encoder/decoder。
- [opencood/extensions/utils.py](opencood/extensions/utils.py)：归一化、mask、量化、ego 特征和重建损失工具。

## 2. 完整调用链

以当前 OPV2V PointPillar multiscale 配置为例，调用链如下：

```text
processed_lidar
  -> PillarVFE
  -> PointPillarScatter
  -> [N, 64, H, W] 的 spatial_features
  -> IBCommunication
  -> 多尺度 BEV backbone
  -> 每个尺度的 AttFusion
  -> decode_multiscale_feature
  -> shrink_header
  -> cls/reg/dir detection heads
  -> point_pillar_loss
```

这里的 `N` 不是普通检测 batch size，而是一个 batch 内所有 CAV 的总数。`record_len` 记录每个场景包含多少个 CAV，例如 `[2, 3]` 表示当前 batch 有两个场景，分别包含 2 和 3 辆车，因此 `N=5`。融合模块使用 `record_len` 和 `pairwise_t_matrix` 将各 CAV 对齐到 ego 车坐标系。

当前配置 [pointpillar_coalign_ib.yaml](opencood/hypes_yaml/back/opv2v/pointpillar_coalign_ib.yaml) 使用：

- voxel size：`[0.4, 0.4, 4]`
- lidar range：`[-140.8, -40, -3, 140.8, 40, 1]`
- 典型 BEV 空间大小：`H=200, W=704`
- 初始特征：`[N, 64, 200, 704]`
- backbone 输出经 shrink 后通常为 `[B, 256, 100, 352]`
- 分类头：`[B, 2, 100, 352]`
- 回归头：`[B, 14, 100, 352]`，因为 `7 * anchor_number` 且 `anchor_number=2`
- 方向头：`[B, 4, 100, 352]`，因为 `2 * anchor_number`

实际尺寸应以真实 batch 为准，不能只依赖代码注释；输入空间尺寸还必须与 decoder 的上采样倍数兼容。

`train_utils.create_model()` 根据 `hypes['model']['core_method']` 动态导入 `opencood.models.<core_method>`，再查找去掉下划线后同名的模型类。因此模型文件名、`core_method` 和模型类名必须匹配。使用 `--model_dir` 时，实际配置通常来自该目录内的 `config.yaml`，所以训练和推理最终以 checkpoint 目录中的配置为准。

## 3. `coib_communication.py` 详细解读

### 3.1 导入和模块职责

[coib_communication.py](opencood/extensions/coib_communication.py) 导入：

- `torch`、`torch.nn`：构建 PyTorch 模块。
- `CoIBEncoder`、`CoIBDecoder`：将空间特征压缩为 latent 并恢复为 BEV 特征。
- `Masknet`、`AdaptiveScaleLayer`、`init_linear_weight`：mask 分支和 latent 参数网络。
- `normalize`、`get_ego_features`、`topk_mask`、`quantize_tensor`、`threshold_sampling`：数据变换、ego 特征覆盖、稀疏选择和量化。

该文件实现的是一个 `nn.Module`，输入是所有 CAV 的 BEV 特征，输出是重建后的 BEV 特征以及 KL/RL 辅助量。

### 3.2 `__init__`

构造函数的重要参数：

- `channels=64`：输入和输出 BEV 通道数。
- `encoding_dim=256`：encoder 输出的 latent 维度。
- `decoding_dim=256`：`mu_net` 和 `std_net` 输出的维度。
- `output_size=(200, 704)`：原始 BEV 空间尺寸。
- `use_transforms=True`：是否先调用 `normalize(x)`。
- `use_ego_feature=True`：是否将 ego 车的原始特征覆盖回输出。
- `out_channels=256`：mask 分支中 `Masknet` 的输出通道数。

```python
self.initial_channels = 256
self.initial_size = (output_size[0] // 8, output_size[1] // 8)
```

decoder 先把 latent 映射为 `[N, 256, H/8, W/8]`，再通过三次 stride=2 的反卷积恢复空间分辨率。因此当前例子为：

```text
输入/输出： [N, 64, 200, 704]
decoder 起点： [N, 256, 25, 88]
第一次反卷积： [N, 128, 50, 176]
第二次反卷积： [N, 128, 100, 352]
第三次反卷积： [N, 64, 200, 704]
```

`output_size` 的高宽应能被 8 整除，否则整数除法和三次上采样可能无法精确恢复原尺寸。

### 3.3 `mu_net`、`std_net` 和信息瓶颈

encoder 输出 `[N, 256]` 后分成两条支路：

- `mu_net` 产生高斯分布均值 `mu`。
- `std_net` 产生标准差 `std`，最后使用 `Softplus` 保证为正，再用 `clamp(min=1e-6)` 防止数值为零。

`AdaptiveScaleLayer` 是可学习的缩放/偏移层。线性层通过 `init_linear_weight` 初始化，bias 初始化为 0。

训练状态下使用重参数化采样：

```text
eps ~ N(0, I)
received_feature = mu + eps * std
```

评估状态下不采样，直接使用：

```text
received_feature = mu
```

因此同一个输入在 `model.train()` 和 `model.eval()` 下的 latent 行为不同。

### 3.4 `forward`

主流程是：

```text
x
 -> get_mask()
 -> autoencoder_forward(x, mask)
 -> 可选地用 ego 原始特征覆盖重建结果
 -> reconstructed_features, KL_value, RL_value
```

`get_ego_features(x, record_len)` 计算每个场景第一个 CAV 的索引。通常这个位置就是 ego 车。若 `use_ego_feature=True`：

```python
reconstructed_features[indices] = ego_features
```

这会保留 ego 车自己的原始 BEV 特征，只对协作者特征使用压缩和重建结果。它依赖 CAV 排列顺序正确，并依赖 `record_len.sum() == x.shape[0]`。

### 3.5 `get_mask`

mask 分支执行：

```text
x
 -> Masknet(64 -> 256)
 -> cls_head(256 -> 2)
 -> ConvTranspose2d(2 -> 1)
 -> sigmoid
 -> topk_mask
 -> 4-bit quantize
```

`cls_head` 在这里与最终检测分类头是同一个模块，但调用位置不同：mask 生成时输入是 `Masknet` 产生的 256 通道特征；最终检测时输入是融合后的 256 通道特征。当前 multiscale 配置必须保证 mask 分支调用时通道和空间尺寸匹配。

#### ratio 调度

如果 `data_dict` 中没有 `epoch` 或 `max_epoch`，代码使用固定 `ratio=0.1`。

如果两者存在，代码计算：

```python
ratio = clamp(linspace(1, -1, max_epoch), min=0.1)[epoch]
```

训练时还会调用 `threshold_sampling` 对 ratio 做随机扰动，评估时不做这一步。由于 `linspace(1, -1)` 随后被 clamp 到 0.1，ratio 后期会在下限附近保持较长时间；这是当前实现行为，若需要改变调度应单独修改并验证。

`topk_mask` 对每个样本展平空间维度，保留 `ceil(H*W*ratio)` 个最大位置，其余位置置零。值相同的边界位置会随机选取，因此训练中 mask 可能有随机性。

#### 量化和梯度

`quantize_tensor(..., bits=4)` 将 `[0,1]` 范围的 mask 量化到 16 个等级，再反量化为浮点数。外层写法：

```python
(quantized - spatial_mask).detach() + spatial_mask
```

使前向使用量化后的值，而反向梯度近似穿过连续的 `spatial_mask`，这是 straight-through estimator。注意 `topk_mask` 的硬选择本身不可导，STE 只为后续连续 mask 路径提供近似梯度。

### 3.6 `autoencoder_forward`

当 `use_transforms=True` 时，先调用 `normalize(x)`。当前 `normalize` 实际是逐元素 `sigmoid`，不是均值/方差标准化；`normalize_data` 虽然存在，但该文件中的主路径没有调用它。

之后：

```text
adapt_x
 -> CoIBEncoder
 -> encoded_feature [N, 256]
 -> mu_net/std_net
 -> latent sampling or mu
 -> CoIBDecoder(latent, mask)
 -> decoded_feature [N, 64, H, W]
```

当前 [coib_encoder.py](opencood/extensions/coib_encoder.py) 中，encoder 是三段 residual block：

- `layer1` 保持空间大小，输出通道约为 32。
- `layer2` stride=2，下采样一次，输出通道约为 16。
- `layer3` stride=2，再下采样一次，输出通道约为 8。
- `AdaptiveMaxPool2d((32,32))` 固定空间大小。
- flatten 后由线性层输出 `[N,256]`。

decoder 使用全连接层展开到 `[N,256,H/8,W/8]`，然后三次反卷积恢复到原尺寸。启用 mask 时，decoder 的每个阶段都会将特征乘以对应的 mask 卷积分支结果：

```text
conv1(x) * mask_conv1(mask)
conv2(x) * mask_conv2(mask)
conv3(x) * mask_conv3(mask)
```

这使 mask 同时影响前向重建和反向梯度。

#### KL loss

当 `required_KL=True` 时，代码计算标准高斯信息瓶颈形式的 KL：

```text
KL = mean(-0.5 * sum(1 + log(std^2) - mu^2 - std^2, dim=1))
```

结果是一个标量，梯度可以回传到 encoder、`mu_net` 和 `std_net`。

#### RL loss

`reconstruct_loss(decoded_feature, x)` 在 `required_RL=True` 时才会计算。该损失对非零目标和零目标使用不同权重，并附加输出 L1 稀疏项。

但是当前 `IBCommunication.forward()` 没有暴露 `required_RL` 参数，调用 `autoencoder_forward()` 时也没有传入 `required_RL=True`，所以当前主训练路径通常返回 `RL_value=None`。`train_infocom.py` 打印 RL，但没有把 RL 加入总损失。

### 3.7 文件底部示例的限制

`coib_communication.py` 的 `__main__` 示例不能直接作为当前模块的 smoke test：

- `IBCommunication.forward()` 需要 `record_len`、`cls_head`、`data_dict`，示例却传入了不存在的 `mask=` 参数。
- `reconstruct_loss` 是从 `extensions.utils` 导入的函数，不是 `model.reconstruct_loss` 成员方法。

应使用完整模型和真实 batch 验证，而不是直接运行该文件底部示例。

## 4. Baseline 与 multiscale 的区别

必须区分以下两条路径：

### 当前配置实际路径

[pointpillar_coalign_ib.yaml](opencood/hypes_yaml/back/opv2v/pointpillar_coalign_ib.yaml) 的配置是：

```yaml
model:
  core_method: point_pillar_baseline_multiscale_ib
```

该模型 [point_pillar_baseline_multiscale_ib.py](opencood/models/point_pillar_baseline_multiscale_ib.py) 导入的是：

```python
from opencood.extensions.coib_communication import IBCommunication
```

通信模块位于 `PointPillarScatter` 之后、BEV backbone 之前，处理典型的 `[N,64,200,704]` 特征，然后才进行多尺度 backbone 和融合。这是用户选中代码的主要调用场景。

### README 示例路径

README 示例建议将模型改为：

```yaml
model:
  core_method: point_pillar_baseline_ib
```

[point_pillar_baseline_ib.py](opencood/models/point_pillar_baseline_ib.py) 导入的是 `coib_communication_baseline.py`，不是当前文件。该 baseline 模型的通信模块调用位置和当前 multiscale 模型不同。

仓库中已经存在文档与配置不一致的情况，因此不要只根据 README 判断实际代码路径。最终应检查：

1. `--model_dir` 指向的目录是否存在 `config.yaml`。
2. `config.yaml` 的 `model.core_method` 是什么。
3. 对应模型文件导入的是 `coib_communication.py` 还是 baseline 版本。

baseline 路径中通信模块固定使用 64 通道，而其调用位置可能接近 backbone/shrink 后的特征，存在通道和空间尺寸不匹配风险。这是需要通过真实前向验证的风险，不应在没有运行的情况下当作确定结论。

## 5. 如何训练 InfoCom

InfoCom 推荐两阶段训练：先训练基础协同感知模型，再加载基础权重训练通信模块。

### 5.1 环境和数据准备

安装方式沿用 OpenCOOD/CoAlign。README 明确指出提供的 checkpoint 依赖 `spconv==1.2.1`。还应确认 PyTorch/CUDA、`open3d`、`tensorboardX`、`cython`、`h5py`、`shapely` 等依赖可用。

当前示例 YAML 使用以下数据目录：

```text
/DATACENTER1/data/opv2v/train
/DATACENTER1/data/opv2v/validate
/DATACENTER1/data/opv2v/test
```

这些是原实验环境路径，必须改成本机真实路径。DAIR-V2X 需要使用 README 指定的补充标注；OPV2V 数据准备遵循 OpenCOOD/CoAlign 的方式。

训练前建议检查：

```bash
cd /data1/bohnsix/InfoCom
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import yaml; print(yaml.safe_load(open('opencood/hypes_yaml/back/opv2v/pointpillar_coalign_ib.yaml'))['model']['core_method'])"
test -d /DATACENTER1/data/opv2v/train
test -d /DATACENTER1/data/opv2v/validate
test -d /DATACENTER1/data/opv2v/test
```

### 5.2 阶段一：训练基础模型

基础模型不带通信效率模块。当前 multiscale CoAlign 基础配置可使用：

```bash
cd /data1/bohnsix/InfoCom
python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/back/opv2v/pointpillar_coalign.yaml
```

开始前先将该 YAML 中的 `root_dir`、`validate_dir`、`test_dir` 改成本机路径。训练完成后，`train.py` 会在 `opencood/logs` 下创建实验目录，通常包含：

- `config.yaml`
- checkpoint，例如 `net_epoch*.pth`
- 验证集最优 checkpoint，例如 `net_epoch_bestval_atN.pth`
- 训练日志和脚本备份

应记录实际实验目录、配置、GPU、依赖版本和数据路径。

### 5.3 阶段二：训练 InfoCom

准备阶段一目录时，至少需要保留：

1. `config.yaml`。
2. 基础模型的 best checkpoint。
3. 与基础模型匹配的数据配置。

然后确认 `config.yaml`：

```yaml
ib_params:
  beta: 0.0001
  begin_epoch: 15

model:
  core_method: point_pillar_baseline_multiscale_ib
```

如果要复现当前 multiscale 实现，应使用 `point_pillar_baseline_multiscale_ib`；如果使用 README 推荐的 baseline 类，则实际导入的是 `coib_communication_baseline.py`，两者不要混淆。

README 示例使用 `beta: 0.001`，当前 `pointpillar_coalign_ib.yaml` 使用 `beta: 1e-4`。两者不同，必须以实际 checkpoint 目录的 `config.yaml` 为准。

基础权重文件名应按当前加载逻辑和 README 约定检查。常见做法是将 best 权重整理为：

```text
net_epoch_bestval_at1.pth
```

不要在没有确认加载器行为的情况下随意删除其他 checkpoint。

启动 InfoCom 训练：

```bash
cd /data1/bohnsix/InfoCom
python opencood/tools/train_infocom.py \
  --model_dir /data1/bohnsix/InfoCom/opencood/logs/<你的实验目录>
```

`train_infocom.py` 的关键行为：

- 从 `<model_dir>/config.yaml` 加载配置和模型。
- 调用 `load_saved_model(..., finetune_flag=True)` 加载基础权重。
- 训练 epoch 不超过 `begin_epoch` 时，优化目标只有 detection task loss。
- 后续 epoch 若 `KL_loss` 存在，使用：

```text
total_loss = task_loss + beta * KL_loss
```

- 训练循环把 `epoch` 和 `max_epoch` 放入 `data_dict`，供 spatial mask ratio 调度。
- 当前 RL reconstruction loss 不会加入总损失。
- 每隔 `eval_freq` 个 epoch 做验证，每隔 `save_freq` 个 epoch 保存普通 checkpoint。
- 验证损失下降时保存 `net_epoch_bestval_atN.pth`。
- 训练结束后脚本会自动调用 inference，因此测试命令和环境必须可用。

当前示例配置的主要训练参数为：

```text
batch_size: 4
epoches: 30
eval_freq: 2
save_freq: 3
optimizer: Adam, lr=0.002
scheduler: MultiStepLR, milestones=[10, 15], gamma=0.1
beta=1e-4
begin_epoch=15
```

实际 batch size、epoch 和学习率应根据显存及实验目的调整；修改后要在配置和实验记录中保持一致。

## 6. 推理与结果验证

InfoCom 训练完成后可以手动运行：

```bash
cd /data1/bohnsix/InfoCom
python opencood/tools/inference.py \
  --model_dir /data1/bohnsix/InfoCom/opencood/logs/<InfoCom实验目录> \
  --fusion_method intermediate
```

`--fusion_method intermediate` 指 OpenCOOD 的推理融合流程，不是用来选择 `coib_communication.py` 的开关。模型类由模型目录中的 `config.yaml` 决定。

建议分别对基础模型和 InfoCom 模型记录：

- IoU 0.3、0.5、0.7 下的 AP/TP/FP。
- checkpoint 实际加载的文件。
- spatial mask 保留比例。
- 单 batch 推理时间和显存。
- 训练与 eval 模式下的输出差异。

开始完整训练前，建议先做一次短实验或真实 batch 验证：

1. `record_len.sum()` 等于输入特征的 CAV 数量。
2. `spatial_features` 是 `[N,64,H,W]`，mask 与 decoder 每一级空间尺寸匹配。
3. encoder 输出是 `[N,256]`，decoder 输出恢复到 `[N,64,H,W]`。
4. `KL_loss` 是有限标量，并在 `begin_epoch` 之后确实进入总损失。
5. `model.train()` 时 latent 使用随机采样，`model.eval()` 时使用 `mu`。
6. ego 索引覆盖每个场景的第一个 CAV，且 ego 特征被正确保留。
7. checkpoint 的 missing/unexpected keys 没有被静默忽略。
8. 基础模型和 InfoCom 模型都能完成 inference。

## 7. 常见问题和风险

- README 中的 `/home/wqm/...`、`/home/user/...` 等旧机器路径不能直接复制，必须改为本机路径。
- `--model_dir` 目录必须包含与 checkpoint 匹配的 `config.yaml`。
- README 推荐 `point_pillar_baseline_ib`，当前 multiscale YAML 使用 `point_pillar_baseline_multiscale_ib`；对应通信文件不同。
- `spconv` 版本与预训练权重兼容性需要确认，README 指定 `spconv==1.2.1`。
- `num_workers=16`、`prefetch_factor=2` 可能在小机器或容器中造成共享内存不足；出现 DataLoader worker 崩溃时应降低它们。
- `epoch` 或 `max_epoch` 缺失时 mask ratio 固定为 0.1，手工调用模型可能和正式训练不一致。
- `topk_mask` 的 ratio 是保留比例，且相同值的边界位置会随机选择。
- decoder 需要输入输出空间尺寸与三次 stride=2 上采样匹配；高宽不能只靠注释推断，应通过真实 batch 验证。
- `required_KL=True` 不代表计算了 RL；当前 InfoCom 训练总损失没有 reconstruction loss。
- checkpoint 加载若使用非严格模式，可能隐藏模型类、配置或权重版本不匹配；应检查加载日志中的 key 差异。
- best checkpoint 的清理和文件名解析依赖 `net_epoch_bestval_atN.pth` 格式，非标准命名需谨慎。
- `train.py` 和 `train_infocom.py` 结束时会自动通过 `os.system` 启动 inference，路径错误会导致训练后最后一步失败。
- `coib_communication.py` 底部 `__main__` 示例已经过时，不能作为模块测试命令。
- 当前工作区存在已有的 C 编译文件、Python 缓存和 `setup.py` 修改；处理训练文档时不要删除或重置这些无关文件。

## 8. 扩展代码时的约束

- 新通信模块应保持模型返回字典中的 `cls_preds`、`reg_preds`、可选 `dir_preds`、`KL_loss` 和 `RL_loss` 契约。
- 修改 `core_method` 时同步检查模型文件名、类名、通信模块导入和 checkpoint 的 `config.yaml`。
- 修改 mask 或 decoder 时，先用一个真实 batch 打印所有中间张量形状，再运行短训练。
- 若要让 RL 真正参与训练，需要同时修改通信模块调用接口和 `train_infocom.py` 的总损失逻辑，并单独验证 loss 的尺度。
- 不要把 README 的历史实验路径、实验参数或 best checkpoint 名称直接当成当前代码的唯一规范；当前配置和实际加载日志优先。
