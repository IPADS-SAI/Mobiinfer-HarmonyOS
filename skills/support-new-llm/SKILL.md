---
name: support-new-llm
description: 为 MNN 框架添加新的 LLM 模型支持。支持从 HuggingFace/ModelScope 下载模型，分析架构，添加映射，Hook 对齐测试，导出 MNN 模型。采用 TDD 模式，分 6 步执行，每步有独立测试标准。
---

# MNN LLM 新模型支持 SKILL

> **触发条件**：当用户请求支持/添加/适配一个新的 LLM 模型时触发。常见表述包括："支持xxx模型"、"添加xxx模型支持"、"适配xxx"、"导出xxx模型"等。

## 概述

本 SKILL 指导 AI Agent 为 MNN 框架添加新的 LLM 模型支持。整个流程采用 **TDD（测试驱动）模式**，每一步都有明确的测试标准，**必须通过当前步骤的测试后才能进入下一步**。

### 核心思路

MNN 的模型导出本质上是**对照 HuggingFace transformers 库中原始模型的实现代码**，将其计算逻辑映射到 MNN 的统一框架中。核心步骤是：

1. 读懂 HF 模型的 `config.json` 和 `modeling_*.py`
2. 在 `model_mapper.py` 中注册字段映射
3. 用 Python `--test` 验证映射正确性
4. 导出 MNN 模型并用 C++ 引擎验证

### 注意事项

> **🚨 严禁将输出错误归因于"量化精度不够"**：4bit 量化的 0.5B 小模型都能正确输出。如果 C++ 输出完全不对（如图片识别不出、输出乱码），**一定是实现细节没有与 HF 对齐**，必须逐步 dump 数据对比定位，不要靠猜。

> **🚨 测试标准要有定力**：每步的通过标准是明确的（如"C++ 能正确描述图片内容"），不能因为"差不多能跑"就跳过。"能感知到一些信号但描述不准确"不等于通过，必须达到与 HF 模型相当的输出质量才算完成。

> **严禁访问以下目录**：`schema/private/` 和 `source/internal/`，包含内部私有代码，**不得读取、修改或引用**。

> **禁止猜测**：如果不确定某个字段名或路径，必须通过工具读取实际文件确认。

> **静态 KV 导出需验证完整调度**：对照原始 HF 整模型检查跨块 prefill、尾块 padding、多步 decode、全部有效 KV，以及多模态位置和特征注入；单层通过或纯 NPU 编译成功均不能替代整模型对照。动态图各档数学正确也不保证厂商能合并编译后的权重/常量缓冲区，应保留单独静态阶段作为可验证基线。

> **压缩 KV 需核对实际存储和误差基线**：包含 scales、保护窗口的字节数，检查图端口真实 dtype 及图内反量化，排除整份浮点影子缓存。使用相同冻结权重/激活状态的浮点 KV 基线，分别报告数值门槛、贪心 token 和实际生成；长上下文超阈值必须保留失败状态，不能以编译成功或短样例相近代替。

> **环形缓存与长输入测试**：保护窗口可能在提交前覆盖旧 token，异常回滚须恢复这些槽位；mask 要验证每个有效历史 token 恰好参与一次。长前缀应重复用户正文并保留一次完整对话模板，将特殊 token 压力测试与普通长输入分开记录。

> **设备接入要验证实际宿主调度**：将端侧分词、图片预处理、M-RoPE、mask、DeepStack 与压缩 KV 更新放入可在主机执行的实现，用独立 HF/ONNX 参考逐输入检查，覆盖跨块图片、不同图片标签顺序、取消和错误后重试。编译器合并输出时任一等价名称都可能保留，端口映射须同时支持合并/未合并接口。App 内复制 tokenizer 等实现应隔离符号，避免改变既有推理库的执行。

> **加载失败先保留驱动调用上下文**：离线图加载日志必须包含图名、文件大小、设备版本和各设置/Build 步骤的独立返回值，不能把选项设置失败统一标成 Build 失败。先定位失败图是否涉及量化，再讨论量化原因；主机测试替身验证调用和清理，不能证明设备支持。首次加载峰值期间尽量延后尚不需要的常驻状态分配。

> **大常量图要单独检查部署预算**：词嵌入等 Gather 表可能比量化后的 decoder 子图更大。Build 前兼容性检查通过不代表资源/鉴权检查通过。若按张量维度分片，应逐值核对原冻结权重、验证多行输出拼接和尾片，并显式要求新运行时和新模型包；公开 NNRT 源码的阈值只能作为设备问题线索，不能当作私有驱动根因已被确认。

> **单图 probe 不覆盖多图常驻**：完整推理要核对每个 executor 和设备缓冲区的释放时机，覆盖多段 prefill、跨块与多步 decode。Build 成功后，executor 创建仍可能因服务调度或资源不足失败；不能用 App RSS 或 CmaFree 排除设备侧占用。采用有界常驻策略时必须在新图 Build 前释放旧资源，确认宿主 KV/输出独立持有，并测试资源上限下的连续执行、错误清理和重试；明确模型重载的性能代价。

> **模型命中与 KV 增量同步要分别验证**：用完整循环调度检查命中率，容量小于工作集时普通 LRU 可能每轮全部失效。独立持有 KV 张量时，复用键需覆盖设备、格式、dtype 和 shape；用缓存世代、源地址及已提交长度决定增量范围，覆盖环形窗口跨界、长跳、重置和错误重试，并逐字节对照完整同步。模型/缓冲区预算不能由文件大小代替实际设备内存；报告主机拷贝量与真实设备性能的边界。

> **宿主特征拷贝前校验完整接口**：配置中的模型维度必须与各阶段逻辑端口的名字、顺序、dtype 和 shape 一致，仅验证单个维度合法或设备端口匹配不够。CPU 参考回放还应独立检查视觉位置常量、特征注入、跨图 hidden 传递和最后有效 token 的选取；直接用宿主提供的全部输入运行参考图会漏掉宿主调度错误。接入新执行模式时检查调试/性能页等所有模型加载入口，避免默认模式或配置复制改变相对资源路径。

> **保留 ONNX 导出版本**：简化器重建 `ModelProto` 时不要无意升级 IR 版本；应核对目标运行时支持范围，并用实际外部权重文件执行导出后的 ONNX。只检查 opset 或文件可解析不足以证明可运行。

> **量化权重与执行参考分开校验**：原生 LUT 索引图只供厂商编译，CPU 对照使用反量化权重和冻结激活量化的参考图。逐值核对 initializer 后再绑定 SDK 算子名；手动展开的 attention 可能丢失模块作用域，二维 Linear 也可能导出为 Gemm。共享 embedding/head 的模型在单独优化 head 时须解除共享，并验证 checkpoint 重载没有改动 embedding。

> **长流程元数据显式 UTF-8 并原子保存**：不要依赖进程 locale，ASCII 默认编码写中文路径会在计算完成后失败。checkpoint、量化参数、图及编译产物均保留哈希关联，避免恢复阶段混用不同候选。

---

## 核心文件清单

| 文件路径 | 作用 | 修改频率 |
|---------|------|---------:|
| `transformers/llm/export/utils/model_mapper.py` | 模型字段映射 | **几乎每个新模型** |
| `transformers/llm/export/utils/model.py` | 统一模型类 `LlmModel` | 偶尔 |
| `transformers/llm/export/utils/transformers.py` | Attention/Decoder/Rotary 等组件 | 新架构时 |
| `transformers/llm/export/utils/config.py` | 模型配置类 `LlmConfig` | 偶尔 |
| `transformers/llm/export/utils/vision.py` | Vision Encoder 实现 | 视觉模型 |
| `transformers/llm/export/utils/audio.py` | Audio Encoder 实现 | 音频模型 |
| `transformers/llm/export/utils/custom_op.py` | 自定义算子导出 | 新算子时 |
| `transformers/llm/export/llmexport.py` | 导出主流程入口 | 偶尔 |

---

## 分步流程总览

整个流程分为 **6 个步骤**，每个步骤都有独立的文档和测试标准：

```
┌──────────────────────────────────────────────────────────┐
│  步骤 1: 下载、理解与测试模型 (step1-analyze.md)           │
│  输入: 模型链接(HF/ModelScope)或本地路径                   │
│  输出: 模型下载到本地 + transformers 推理成功               │
│       + 架构分析完成 + Tier 判定                           │
│  测试: test_origin.py 输出正确 + 5 个架构问题已回答        │
├──────────────────────────────────────────────────────────┤
│  步骤 2: 添加映射 (step2-mapping.md)                      │
│  输入: 步骤1的差异清单                                     │
│  输出: model_mapper.py 中的新映射                          │
│  测试: LlmModel.from_pretrained 加载不报错                 │
├──────────────────────────────────────────────────────────┤
│  步骤 3: Hook 对齐测试 (step3-test-python.md)             │
│  输入: 步骤1的原始模型结果 + 步骤2的 LlmModel              │
│  输出: Hook 中间结果数值对齐                               │
│  测试: 5 个检查点(embed/layer0/lastlayer/norm/logits)一致  │
├──────────────────────────────────────────────────────────┤
│  步骤 4: 导出与 C++ 测试 (step4-export.md)                │
│  输入: 步骤3通过                                           │
│  输出: MNN 模型文件                                        │
│  测试: C++ llm_demo 输出正确                               │
├──────────────────────────────────────────────────────────┤
│  步骤 5: 视觉/音频支持 (step5-multimodal.md)              │
│  输入: 仅 Tier 4/5/6 需要                                  │
│  输出: vision.py 或 audio.py 中的新子类                    │
│  测试: 多模态推理测试通过                                   │
├──────────────────────────────────────────────────────────┤
│  步骤 6: 特殊架构支持 (step6-new-architecture.md)         │
│  输入: 仅 Tier 6 需要                                      │
│  输出: 新算子 + C++ 实现                                   │
│  测试: 全链路测试通过                                       │
└──────────────────────────────────────────────────────────┘
```

### 步骤选择指南

**不是所有步骤都需要执行。** 根据步骤 1 判定的 Tier，选择需要执行的步骤：

| Tier | 需要执行的步骤 | 说明 |
|------|--------------|------|
| Tier 1 (纯文本 Llama-like) | 1 → 2 → 3 → 4 | 最简单，仅需映射 |
| Tier 2 (轻微架构差异) | 1 → 2 → 3 → 4 | 可能需要修改 transformers.py |
| Tier 3 (MoE 模型) | 1 → 2 → 3 → 4 | 需要 mlp/expert 映射 + routing 实现，参见 `common-pitfalls.md` 第 9 节 |
| Tier 4 (音频模型) | 1 → 2 → 3 → 5 → 4 | 需要 audio.py |
| Tier 5 (视觉模型) | 1 → 2 → 3 → 5 → 4 | 需要 vision.py |
| Tier 6 (全新架构) | 1 → 2 → 6 → 3 → 4 | 需要新算子（如叠加 Tier 4/5 则加入 step5） |

---

## Tier 判定速查

根据 `config.json` 中的字段快速判定 Tier：

```
config.json 中是否有 num_experts 或 num_local_experts?
├─ 是 → Tier 3 (MoE)
└─ 否 → 继续

config.json 中是否有 vision_config?
├─ 是 → Tier 5 (视觉)
└─ 否 → 继续

config.json 中是否有 audio_config?
├─ 是 → Tier 4 (音频)
└─ 否 → 继续

layer_types 中是否有非 Attention 层（如 conv / mamba / rwkv）?
├─ 是 → Tier 6 (混合架构，如 lfm2 的 conv + full_attention)
└─ 否 → 继续

modeling_*.py 中是否有全新的 Attention 类型（非标准 SDPA）?
├─ 是 → Tier 6 (新架构，如 qwen3_5 的 gated_delta_rule)
└─ 否 → 继续

是否有额外 LayerNorm / scale_depth / scale_emb / 没有 post_attention_layernorm?
├─ 是 → Tier 2 (轻微差异)
└─ 否 → Tier 1 (标准 Llama-like)
```

> **注意**：Tier 可以叠加。例如 MoE + 视觉 = Tier 3+5，需要同时执行对应步骤。

### Tier 叠加时的执行顺序

当模型跨多个 Tier 时，按以下原则确定步骤顺序：

| 叠加 | 执行顺序 | 说明 |
|------|---------|------|
| Tier 3+6 (MoE + 新架构) | 1 → 2 → 6 → 3 → 4 | 先支持新架构（如 short_conv），再加 MoE routing |
| Tier 3+5 (MoE + 视觉) | 1 → 2 → 3 → 5 → 4 | MoE 是 text-only，视觉单独加 |
| Tier 5+6 (视觉 + 新架构) | 1 → 2 → 6 → 5 → 3 → 4 | 先支持新架构，再加视觉 |

**原则**：Tier 6（新架构）最先实现，因为它影响基础层结构；Tier 3（MoE）次之，因为它在 Tier 6 层结构之上；Tier 4/5（多模态）最后，因为它与层结构独立。

---

## 已支持模型速查表

收到新模型请求时，先查 `config.json` 的 `model_type`，在下表中搜索。如果已存在则**无需修改**。

| model_type | Tier | 类型 |
|-----------|------|------|
| `llama`, `qwen2`, `internlm`, `mobilellm` | 1 | 文本 |
| `baichuan` | 1 | 文本 (fused QKV) |
| `qwen` | 1 | 文本 (Qwen1) |
| `qwen3` | 1 | 文本 (q/k norm) |
| `chatglm`, `chatglm2` | 1 | 文本 (特殊 RoPE) |
| `phi-msft`, `phi` | 1 | 文本 |
| `gemma2` | 2 | 文本 (额外 LayerNorm) |
| `gemma3_text` | 2 | 文本 |
| `minicpm` | 2 | 文本 (scale_depth) |
| `gpt_oss` | 3 | MoE (topk → softmax) |
| `qwen3_moe` | 3 | MoE (softmax → topk) |
| `lfm2_moe` | 3+6 | MoE (sigmoid routing) + 混合架构 (short_conv) |
| `qwen2_audio` | 4 | 音频 |
| `funaudiochat` | 4 | 音频 |
| `lfm2_audio` | 4+6 | 音频 (FastConformer + MLP adapter) + 混合架构 |
| `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl` | 5 | 视觉 |
| `internvl_chat` | 5 | 视觉 |
| `gemma3` | 5 | 视觉 |
| `gemma4` | 5 | 视觉 (PLE + KV共享 + 双Rotary + patch-based vision) |
| `glm_ocr` | 5 | 视觉 (Gemma2残差 + interleaved M-RoPE) |
| `smolvlm` | 5 | 视觉 (SigLIP + Perceiver) |
| `idefics3` | 5 | 视觉 (SigLIP + Perceiver) |
| `lfm2_vl` | 5+6 | 视觉 (SigLIP2 NaFlex + pixel_unshuffle) + 混合架构 |
| `lfm2` | 6 | 混合架构 (short_conv + full_attention) |
| `qwen3_5` | 6 | 视觉+LinearAttn (gated_delta_rule) |

---

## 常见陷阱

**在开始之前，建议先浏览 `common-pitfalls.md`**，了解已知的常见问题和解决方案（RoPE 变体、dtype 级联、Jinja 限制、stop token、残差模式、MoE 支持要点、FakeLinear axis 陷阱、**do_map 静默失败与 rope_theta 间接存储**、非标准模型加载等）。

---

## 开始执行

**现在请打开 `skills/support-new-llm/step1-analyze.md`，开始步骤 1。**
