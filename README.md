# 羽人-百川 7B

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Lint](https://github.com/pleisto/yuren-baichuan-7b/actions/workflows/lint.yml/badge.svg)](https://github.com/pleisto/yuren-baichuan-7b/actions/workflows/lint.yml) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE) [![huggingface badge](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-f2f4f5?labelColor=f2f4f5)](https://huggingface.co/pleisto/yuren-baichuan-7b)

羽人是基于 [baichuan-inc/baichuan-7B](https://huggingface.co/baichuan-inc/baichuan-7B) 进行多任务有监督微调的开源多模态大语言模型, 建立在 [Pleisto](https://github.com/pleisto) 的以数据为中心(Data-centric AI)的工作上。羽人在多轮对话、开放域问答、角色扮演、文本生成、文本理解、图片理解等多个任务上均拥有优异的表现。

YuRen is a multi-modal large language model based on [baichuan-inc/baichuan-7B](https://huggingface.co/baichuan-inc/baichuan-7B) and trained with multi-task supervised fine-tuning. It is built on top of [Pleisto](https://github.com/pleisto)'s data-centric AI work. YuRen has excellent performance on multi-turn dialogue, open-domain question answering, role-playing, text generation, text understanding, image understanding and other tasks. For more english information, please refer to [model card](https://huggingface.co/pleisto/yuren-baichuan-7b).

## 主要亮点

> 💡 **推理效果的演示可以参考[这里](./docs/showcases.md)**

- **多模态**: 参考[LLaVA](https://github.com/haotian-liu/LLaVA) 和 [mPLUG-Owl](https://arxiv.org/abs/2304.14178) 的相关工作, 羽人通过建立线性投影层将 LLM 的语言模态和目前最 SOTA 的 CLIP 模型[laion/clip-vit-l-14-datacomp.xl-s13b-b90k](https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K) 的视觉编码器进行融合, 从而实现了卓越的图片理解能力。
- **超高质量 SFT 数据集**: 羽人的 SFT 数据集的基础数据来自于 Pleisto 自有的商业多轮对话与指令精调数据集的一个子集, 该数据集的所有指令均经过了多轮次的人工和算法质检, 在此基础上我们还参考了[Orca LLM](https://arxiv.org/abs/2306.02707)的工作在该子集上进行了基于 GPT-4 的数据增强。图像模态的数据集则由公共数据集 coco2017、ScienceQA 的子集、laion5b 的子集以及 Pleisto 自有的扩散模型训练数据集的中文子集共同构成。
- **商业友好**: 羽人的训练和推理代码以 Apache-2.0 协议开源, 模型权重的授权则完全继承自[baichuan-7B 模型许可协议](https://huggingface.co/baichuan-inc/baichuan-7B/blob/main/baichuan-7B%20%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf) 仅需联系 [baichuan 团队](opensource@baichuan-inc.com) 进行免费登记即可获得商业使用授权。
- **全面兼容 ChatML**: 羽人全面兼容 GPT-4 同款的[ChatML 格式](https://github.com/openai/openai-python/blob/main/chatml.md), 一方面可以最大限度地减少 Prompt Injection 所带来的安全风险, 另一方面可以和 GPT-4 一样实现良好的 System Prompt 遵循度。(没错, 我们的训练数据集中包含了相当一部分带有 system prompt 的对话数据)

## 使用 WebUI 进行推理

### Docker

> Coming soon

### 本地运行

```bash
# 使用 rye 进行环境管理, 可访问 https://rye-up.com/guide/installation/#installing-rye 查看详情
curl -sSf https://rye-up.com/get | bash
source "$HOME/.rye/env"
rye sync
rye run webui "pleisto/yuren-baichuan-7b" # --load_8bit True --server_name "0.0.0.0" --share True
```

## 局限性

- 受限于较小的参数量，羽人-百川 7B 在数值计算、逻辑推理类任务的效果不尽人意，同时在多模态任务上也无法完全发挥出 CLIP 的优势，存在一定的幻觉现象。如果您有业务场景的真实需求，可以与我们联系，我们还有更大参数量的闭源模型可以提供。未来，我们也会考虑开源更大参数量的模型。
- 当前版本的羽人-百川 7B 尚未经过人类偏好对齐，在输出内容上存在一定的随机性，同一问题的多次回答可能在性能上有明显的差异，后续我们将提供经过人类偏好对齐的模型，以提升模型的稳定性。
- 尽管我们已在训练数据和预置的 System Prompt 层面上进行了内容安全的控制，但模型仍然可能会产生偏见、歧视、虚构或不当的内容，我们强烈建议您在使用模型时采取额外的安全措施，例如对模型的输入输出进行过滤、审查或限制，以避免对您的用户造成伤害。

## 训练数据

遗憾的是, 由于羽人的训练数据集建立在我们的商业数据集的子集之上, 因此我们现阶段没有将其完整开源的计划。目前我们只能提供一个[包含 300 条训练数据的样例数据集](./data/sft.dev.json), 该数据集的格式和我们的完整数据集完全一致, 但是由于数据量太少, 无法训练出一个完整的模型, 仅供大家参考。该样例数据集以[CC BY-SA 4.0 (署名且以相同方式共享)](https://creativecommons.org/licenses/by-sa/4.0/deed.zh-Hans) 协议开源, 详见文件内的`__comment__`字段。


## 评测结果

> 评测环境： [OpenCompass 20230706](https://github.com/internLM/OpenCompass/)， 使用 zero-shot CoT (Chain-of-Thought) 方法测试

作为多模态模型羽人7b在一部分参数量被VQA（视觉问答）任务占用的情况下，依然在纯文本模态领域取得了较好的评测成绩，目前在主流的7B LLM中占第三名。值得注意的是这还是在羽人7b仅进行了SFT，没有进行进一步的RLHF对齐的情况下取得的成绩。

### MMLU 英文评测

| Model                                  | Humanities | Social Sciences | STEM | Other | Average |
|----------------------------------------|-----------:|:---------------:|:----:|:-----:|:-------:|
| LLaMA-7B<sup>2</sup>                   |       34.0 |      38.3       | 30.5 | 38.1  |  35.1   |
| Falcon-7B<sup>1</sup>                  |          - |        -        |  -   |   -   |  35.0   |
| mpt-7B<sup>1</sup>                     |          - |        -        |  -   |   -   |  35.6   |
| ChatGLM-6B<sup>0</sup>                 |       35.4 |      41.0       | 31.3 | 40.5  |  36.9   |
| BLOOM 7B<sup>0</sup>                  |       25.0 |      24.4       | 26.5 | 26.4  |  25.5   |
| BLOOMZ 7B<sup>0</sup>                 |       31.3 |      42.1       | 34.4 | 39.0  |  36.1   |
| moss-moon-003-base (16B)<sup>0</sup>   |       24.2 |      22.8       | 22.4 | 24.4  |  23.6   |
| moss-moon-003-sft (16B)<sup>0</sup>    |       30.5 |      33.8       | 29.3 | 34.4  |  31.9   |
| Baichuan-7B<sup>0</sup>           |       38.4 |      48.9       | 35.6 | 48.1  |  42.3   |
| **羽人-baichuan-7b-多模态**           |       **41.77** |      **53.97**       | **39.90** | **41.33**  |  **44.24**   |
| chatglm2-6b | 41.23 | 51.61	| 40.06 | 51.24 | 45.46 |
| InternLM-Chat-7b | - | - | - | - | 50.8	|

### C-Eval 中文评测

|        Model         | Average | Avg(Hard) | STEM  | Social Sciences | Humanities | Others |
| :-------------------------: | :-----: | :-------: | :---: | :-------------: | :--------: | :----: |
|            GPT-4            |  68.7   |   54.9    | 67.1  |      77.6       |    64.5    |  67.8  |
|           ChatGPT           |  54.4   |   41.4    | 52.9  |      61.8       |    50.9    |  53.6  |
|         Claude-v1.3         |  54.2   |   39.0    | 51.9  |      61.7       |    52.1    |  53.7  |
|     Claude-instant-v1.0     |  45.9   |   35.5    | 43.1  |      53.8       |    44.2    |  45.4  |
|          BLOOMZ-7B          |  35.7   |   25.8    | 31.3  |      43.5       |    36.6    |  35.6  |
|         ChatGLM-6B          |  34.5   |   23.1    | 30.4  |      39.6       |    37.4    |  34.5  |
|   Ziya-LLaMA-13B-pretrain   |  30.2   |   22.7    | 27.7  |      34.4       |    32.0    |  28.9  |
|  moss-moon-003-base (16B)   |  27.4   |   24.5    | 27.0  |      29.1       |    27.2    |  26.9  |
|         LLaMA-7B-hf         |  27.1   |   25.9    | 27.1  |      26.8       |    27.9    |  26.3  |
|          Falcon-7B          |  25.8   |   24.3    | 25.8  |      26.0       |    25.8    |  25.6  |
|      TigerBot-7B-base       |  25.7   |   27.0    | 27.3  |      24.7       |    23.4    |  26.1  |
|    Aquila-7B<sup>*</sup>    |  25.5   |   25.2    | 25.6  |      24.6       |    25.2    |  26.6  |
| Open-LLaMA-v2-pretrain (7B) |  24.0   |   22.5    | 23.1  |      25.3       |    25.2    |  23.2  |
|          BLOOM-7B           |  22.8   |   20.2    | 21.8  |      23.3       |    23.9    |  23.3  |
|       Baichuan-7B       |  42.8   |   31.5    | 38.2  |      52.0       |    46.2    |  39.3  |
|       **羽人-baichuan-7b-多模态**       |  44.97   |   31.7    | 47.04  |      52.0       |    43.08    |  45.31  |
| chatglm2-6b | 50.1 | -	| 46.4	 | 60.4 | 50.6 | 46.9 |
| InternLM-Chat-7b | 53.2 | - | - | - | -	| - |


## 复现训练

### 准备基座模型

为了兼容 ChatML 格式以及支持图像模态, 我们需要在基座模型中添加几个 Special Token:

```bash
rye sync
rye run prepare-base-model
```

(注:词表大小会被扩充至最接近的 128 的倍数以改善并行训练时的性能)

同时为了便于直接复用 LLaMA 的生态, 我们使用了 LLaMA 兼容的 BaiChuan 权重而非原始权重.

### SFT - Stage 1

不同于纯文本 LLM, 为了避免灾难性遗忘, 我们在第一阶段仅进行 1 个 epoch 的 FineTuning, 第三阶段会再和多模态数据一起进行 2 个 epoch 的 FineTuning。

> 下述脚本均适用于 8 卡 A100 80G 环境, 如需在其他环境下运行请酌情调整相关参数。

初始化环境:

```bash
. .venv/bin/activate
wandb login # 登录 wandb 以便于记录训练日志
```

#### 全量微调

```bash
torchrun --nproc_per_node=8 -m yuren_trainer.text_sft \
  --model_name_or_path "dist/yuren-7b-base" --train_file 'train.json' \
  --validation_file 'validation.json' --model_max_length 4096 \
  --num_train_epochs 1 --per_device_eval_batch_size 4 --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 --evaluation_strategy "steps" --eval_steps 340 \
  --save_strategy "steps" --save_steps 340 --save_total_limit 8 --learning_rate 2e-5 \
  --weight_decay 0. --lr_scheduler_type "cosine" --logging_steps 10 \
  --run_name yuren-llama-7b-stage1 --warmup_ratio 0.03 \
  --dataloader_drop_last True --group_by_length True --tf32 True --bf16 True \
  --deepspeed "apps/yuren_trainer/config/deepspeed_config.json" --output_dir "dist/yuren-7b-stage1"
```

#### QLora

> :warning: 即便是参考的[guanaco](https://arxiv.org/abs/2305.14314)工作将 rank 设置为 `64` 的情况下, Lora 的性能依然不如全量微调, 因此我们通常仅使用全量微调, QLora 仅作为一个低资源下的备选方案。

```bash
torchrun --nproc_per_node=8 -m yuren_trainer.text_sft \
  --model_name_or_path "dist/yuren-7b-base" --train_file 'train.json' \
  --validation_file 'validation.json' --model_max_length 4096 \
  --num_train_epochs 1 --per_device_eval_batch_size 4 --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 --evaluation_strategy "steps" --eval_steps 340 \
  --save_strategy "steps" --save_steps 340 --save_total_limit 8 --learning_rate 5e-5 \
  --weight_decay 0 --lr_scheduler_type "cosine" --logging_steps 4 --tf32 True --bf16 True \
  --run_name yuren-llama-7b-qlora-stage1 --warmup_ratio 0.03 --gradient_checkpointing True \
  --dataloader_drop_last True --group_by_length True --optim "paged_adamw_8bit" --max_grad_norm 0.5 \
  --use_lora True --lora_config "apps/yuren_trainer/config/qlora.json" --output_dir "dist/yuren-7b-stage1"
```

### SFT - Stage 2

> :warning: 现阶段的 Stage 2 和 Stage 3 的训练使用 Pleisto 内部的 monorepo 中的分布式训练脚本进行, 其底层依赖了一些内部类库, 导致暂时无法直接移植开源, 我们会在后续的版本中将其重构并移入本仓库。如果需要复现 Stage 2 和 Stage 3 的训练, 目前建议直接参考 [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main/llava) 的相关代码, 由于 LLaVA 支持基于 [MPT](https://huggingface.co/mosaicml/mpt-7b-chat) 模型的多模态训练, 而 MPT 和羽人一样使用了 ChatML 格式, 因此理论上只需将数据集预处理相关的代码强制设置为 MPT 格式同时维持 LLaMA 模型结构不变即可与羽人兼容。需要注意的是, 我们内部的训练脚本尽管也参考了 LLaVA 但并不完全一致(我们的内部实现还参考了一部分 mPLUG-Owl 的工作), 因此我们无法保证直接使用 LLaVA 的代码能够完全复现我们的结果。

本阶段将冻结 LLM 的权重, 单独训练 Clip 模型中 Vision Encoder 和 LLM 连接所用的线性投影层。

### SFT - Stage 3

本阶段将使用多模态数据集进行 2 个 epoch 的 FineTuning, 同时训练线性投影层和 LLM 本身。 多模态数据集即包括了图片模态的数据集, 也包括了 Stage 1 中使用的纯文本数据集。
