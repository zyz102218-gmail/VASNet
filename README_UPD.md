# 运行环境修正说明
## 摘要
### 适配新版ortools
VASNET这个库的版本比较老旧，其中使用了谷歌的<code>ortools</code>库来作为常用算法的直接调用的集合。这个库更新非常频繁，其中许多语句写法更新很快。所幸其中的算法并没有减少，只是修改了名称。我参考其官方文档，重写了<code>knapsack.py</code>中的部分函数用法，使其在当前主流环境下可以运行，且功能上没有问题。

### 多平台运行与训练适配
Torch是具备跨版本能力的，经过测试我们发现现版本torch可以正常加载模型并实现推理，因此对torch的代码改动很少。

结合近期torch集成的多类后端API自动适应能力，我们对main中的部分代码进行了修改，添加了自适应平台能力。目前支持NVIDIA CUDA, AMD ROCm（未测试）和Apple Silicon MPS平台对模型进行推理和训练。值得注意的是，因为 PyTorch 将 ROCm 作为其 CUDA API 的一个后端实现，所以代码中仍然使用 "cuda" 来代表 AMD GPU。因此在代码中看不到ROCm的具体实现。这一部分修改在main_multiPlatform.py中，并没有直接对main.py进行修改。

但值得注意的是，不同模型对不同平台的适应性可能有差异。VASNET本身是一个较为老旧的模型，其使用的torch特性也已经相对稳定，而新的模型可能会使用新的torch特性，某些torch特性在不同平台上可能有不同实现，甚至没有实现。因此需要特别注意，本工作的内容不一定适用于其他模型的运行与移植。


参考文档：[ortools-官方文档](https://or-tools.github.io/docs/pdoc/ortools/algorithms/python/knapsack_solver.html#KnapsackSolver)

## 测试环境
```bash
ackages & system versions:
----------------------------------------------------------------------
display :  	NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.67  Tue Mar 12 23:54:15 UTC 2024
	GCC version:  gcc version 13.2.1 20240316 (Red Hat 13.2.1-7) (GCC)
cuda :  NA
cudnn :  8902
platform :  Linux-6.8.5-201.fc39.x86_64-x86_64-with-glibc2.38
python :  (3, 10, 14)
torch :  2.2.2
numpy :  1.26.4
h5py :  3.11.0
json :  2.0.9
ortools :  9.9.3963
torchvision :  0.17.2
```

手动整理：
```bash
System Information
CUDA: 12.1
cuDNN: 9.1.0.70_cuda12
GCC: gcc (GCC) 13.2.1 20240316 (Red Hat 13.2.1-7)
G++: g++ (GCC) 13.2.1 20240316 (Red Hat 13.2.1-7)
python: Python 3.10.14

Python Packages
torch              2.2.2
torchaudio         2.2.2
torchvision        0.17.2
numpy              1.26.4
ortools            9.9.3963
```
## 运行结果
```bash
Packages & system versions:
----------------------------------------------------------------------
display :  	NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.67  Tue Mar 12 23:54:15 UTC 2024
	GCC version:  gcc version 13.2.1 20240316 (Red Hat 13.2.1-7) (GCC)
cuda :  NA
cudnn :  8902
platform :  Linux-6.8.5-201.fc39.x86_64-x86_64-with-glibc2.38
python :  (3, 10, 14)
torch :  2.2.2
numpy :  1.26.4
h5py :  3.11.0
json :  2.0.9
ortools :  9.9.3963
torchvision :  0.17.2

Parameters:
----------------------------------------------------------------------
[0] cuda_device: 0
[1] datasets: ['datasets/eccv16_dataset_summe_google_pool5.h5', 'datasets/eccv16_dataset_tvsum_google_pool5.h5', 'datasets/eccv16_dataset_ovp_google_pool5.h5', 'datasets/eccv16_dataset_youtube_google_pool5.h5']
[2] epochs_max: 300
[3] l2_req: 1e-05
[4] lr: [5e-05]
[5] lr_epochs: [0]
[6] max_summary_length: 0.15
[7] output_dir: data
[8] root: 
[9] splits: ['splits/tvsum_splits.json', 'splits/summe_splits.json', 'splits/tvsum_aug_splits.json', 'splits/summe_aug_splits.json']
[10] train: False
[11] train_batch_size: 1
[12] use_cuda: True
[13] verbose: False



Setting CUDA device:  0
Loading: datasets/eccv16_dataset_summe_google_pool5.h5
Loading: datasets/eccv16_dataset_tvsum_google_pool5.h5
Loading: datasets/eccv16_dataset_ovp_google_pool5.h5
Loading: datasets/eccv16_dataset_youtube_google_pool5.h5
Loading splits from:  splits/tvsum_splits.json
Selecting split:  0
Loading model: data/models/tvsum_splits_0_0.6133250787356799.tar.pth
Avg F-score:  0.6133250787356799

Selecting split:  1
Loading model: data/models/tvsum_splits_1_0.6353969877811061.tar.pth
Avg F-score:  0.6353969877811061

Selecting split:  2
Loading model: data/models/tvsum_splits_2_0.5867147987152299.tar.pth
Avg F-score:  0.5867147987152299

Selecting split:  3
Loading model: data/models/tvsum_splits_3_0.6417678540081985.tar.pth
Avg F-score:  0.6417678540081985

Selecting split:  4
Loading model: data/models/tvsum_splits_4_0.5941821875878188.tar.pth
Avg F-score:  0.5941821875878188

Total AVG F-score:  0.6142773813656067


Setting CUDA device:  0
Loading: datasets/eccv16_dataset_summe_google_pool5.h5
Loading: datasets/eccv16_dataset_tvsum_google_pool5.h5
Loading: datasets/eccv16_dataset_ovp_google_pool5.h5
Loading: datasets/eccv16_dataset_youtube_google_pool5.h5
Loading splits from:  splits/summe_splits.json
Selecting split:  0
Loading model: data/models/summe_splits_0_0.47570115033520377.tar.pth
Avg F-score:  0.47570115033520377

Selecting split:  1
Loading model: data/models/summe_splits_1_0.4532441838099789.tar.pth
Avg F-score:  0.4532441838099789

Selecting split:  2
Loading model: data/models/summe_splits_2_0.46392187092332515.tar.pth
Avg F-score:  0.46392187092332515

Selecting split:  3
Loading model: data/models/summe_splits_3_0.5637243258371596.tar.pth
Avg F-score:  0.5637243258371596

Selecting split:  4
Loading model: data/models/summe_splits_4_0.5249340590955511.tar.pth
Avg F-score:  0.5249340590955511

Total AVG F-score:  0.4963051180002437


Setting CUDA device:  0
Loading: datasets/eccv16_dataset_summe_google_pool5.h5
Loading: datasets/eccv16_dataset_tvsum_google_pool5.h5
Loading: datasets/eccv16_dataset_ovp_google_pool5.h5
Loading: datasets/eccv16_dataset_youtube_google_pool5.h5
Loading splits from:  splits/tvsum_aug_splits.json
Selecting split:  0
Loading model: data/models/tvsum_aug_splits_0_0.6328304729233908.tar.pth
Avg F-score:  0.6328304729233908

Selecting split:  1
Loading model: data/models/tvsum_aug_splits_1_0.6059793418308115.tar.pth
Avg F-score:  0.6059793418308115

Selecting split:  2
Loading model: data/models/tvsum_aug_splits_2_0.6197595383836808.tar.pth
Avg F-score:  0.6197595383836808

Selecting split:  3
Loading model: data/models/tvsum_aug_splits_3_0.639623427142229.tar.pth
Avg F-score:  0.639623427142229

Selecting split:  4
Loading model: data/models/tvsum_aug_splits_4_0.624663257236588.tar.pth
Avg F-score:  0.624663257236588

Total AVG F-score:  0.62457120750334


Setting CUDA device:  0
Loading: datasets/eccv16_dataset_summe_google_pool5.h5
Loading: datasets/eccv16_dataset_tvsum_google_pool5.h5
Loading: datasets/eccv16_dataset_ovp_google_pool5.h5
Loading: datasets/eccv16_dataset_youtube_google_pool5.h5
Loading splits from:  splits/summe_aug_splits.json
Selecting split:  0
Loading model: data/models/summe_aug_splits_0_0.5122454899922037.tar.pth
Avg F-score:  0.5122454899922037

Selecting split:  1
Loading model: data/models/summe_aug_splits_1_0.5858084051379817.tar.pth
Avg F-score:  0.5858084051379817

Selecting split:  2
Loading model: data/models/summe_aug_splits_2_0.47047338207156864.tar.pth
Avg F-score:  0.47047338207156864

Selecting split:  3
Loading model: data/models/summe_aug_splits_3_0.49019728744247376.tar.pth
Avg F-score:  0.49019728744247376

Selecting split:  4
Loading model: data/models/summe_aug_splits_4_0.4967810509407774.tar.pth
Avg F-score:  0.4967810509407774

Total AVG F-score:  0.511101123117001

Final Results:
------------------------------------------------------
  No   Split                                Mean F-score
======================================================
  1    splits/tvsum_splits.json             61.428% 
  2    splits/summe_splits.json             49.631% 
  3    splits/tvsum_aug_splits.json         62.457% 
  4    splits/summe_aug_splits.json         51.11%  
------------------------------------------------------
```

## 修改代码整理
修改了<code>kanpsack.py</code>：

导入库部分：
```python
# from ortools.algorithms import pywrapknapsack_solver      # 旧版本用法
from ortools.algorithms.python import knapsack_solver as pywrapknapsack_solver # 解决版本问题
from ortools.algorithms.python.knapsack_solver import SolverType
```

函数实现部分：
```python
osolver = pywrapknapsack_solver.KnapsackSolver(
    # pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
    SolverType.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
    'test')

def knapsack_ortools(values, weights, items, capacity ):
    scale = 1000
    values = np.array(values)
    weights = np.array(weights)
    # values = (values * scale).astype(np.int)
    values = (values * scale).astype(int) # np.int已经废弃，现在使用int类型即可
    # weights = (weights).astype(np.int)
    weights = (weights).astype(int)
    capacity = capacity

    # osolver.Init(values.tolist(), [weights.tolist()], [capacity]) # 旧版本
    osolver.init(values.tolist(), [weights.tolist()], [capacity])
    # computed_value = osolver.Solve()
    computed_value = osolver.solve()
    # packed_items = [x for x in range(0, len(weights))
    #                 if osolver.BestSolutionContains(x)] # 旧版本
    packed_items = [x for x in range(0, len(weights))
                    if osolver.best_solution_contains(x)]

    return packed_items

```
其中注释掉的部分是原本的内容。