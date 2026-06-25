# ContextScope PPT Outline

## 1. Title

Title: ContextScope: Node-Adaptive BoundaryGNN for Social Boundary-Risk Detection

Subtitle: 基于 SNAP Google+ Circles 的社交边界风险节点识别

Key message: 本项目把公开社交圈数据重新定义为边界风险识别任务，并提出节点级动态门控 BoundaryGNN。

Visual: 简洁标题页，背景可用社交网络节点图或方法结构局部图。

## 2. Motivation

Title: 为什么不是社区发现，而是边界风险识别？

Key message:
- 社交网络中的风险不只来自“用户属于哪个圈层”，也来自“用户是否连接多个圈层”。
- 一个跨圈层节点可能造成信息从一个 audience 泄露到另一个 audience。
- 因此目标从 circle discovery 转为 boundary-risk node detection。

Visual: 左侧画三个社交圈，右侧突出一个跨圈层桥接节点。

## 3. Dataset and Task

Title: Public Dataset and Reconstructed Task

Key message:
- Dataset: SNAP Google+ Circles。
- 原始数据包含 ego-network edges、匿名画像特征、circle 标注。
- 本项目重新构造弱监督二分类任务：预测节点是否为边界风险节点。

Data:
- 132 raw ego networks。
- 122 valid ego networks。
- 254,556 node instances。
- 26,954,330 edge instances。
- 39,204 labeled nodes。
- Train / Val / Test = 23,523 / 7,847 / 7,834。

Visual: 数据规模统计表或四个关键数字卡片。

## 4. Label Construction

Title: Weak Boundary-Risk Labels from Circle Annotations

Key message:
- 正样本不是人工标注，而是由 circle membership 和跨圈层邻居暴露构造。
- 一个节点属于多个 circle，或连接多个外部 circle 且外部邻居比例较高，则视为 boundary-risk。
- 验证集只用于阈值选择，不参与训练。

Formula:
boundary_risk(v) = 1 if multi-circle membership or strong outside-circle exposure.

Visual: 一个节点的 own circle、outside circle、outside neighbor ratio 示意图。

## 5. Baselines

Title: Baselines: Keep Comparison Focused

Key message:
- 只保留 3 个清晰 baseline，避免对比过散。
- ProfileLogit: 只看画像。
- StructureLogit: 只看结构。
- ProfileStructureLogit: 画像 + 结构，但不做邻居聚合。

Visual: 三个 baseline 小模块并排，下面连接到同一个 binary classifier。

## 6. BoundaryGNN-Logit

Title: Boundary-Aware Neighbor Aggregation

Key message:
- 基于画像 Jaccard similarity 对好友边分型。
- 分别聚合 all-neighbor、similar-neighbor、dissimilar-neighbor。
- 拼接 self profile 和 structural features。

Representation:
h_v = concat(self_v, all_v, similar_v, dissimilar_v, structure_v)

Visual: 方法结构图，五个通道进入 classifier。

## 7. Main Result

Title: Boundary-Aware Aggregation Improves over Simple Baselines

Key message:
- BoundaryGNN-Logit 在 F1 和 AUC 上超过三个轻量 baseline。
- 说明边界风险识别不能只靠画像或结构，需要边界感知邻居上下文。

Table:
- BoundaryGNN-Logit: F1 0.7379, AUC 0.7592。
- StructureLogit: F1 0.7140, AUC 0.7199。
- ProfileStructureLogit: F1 0.7041, AUC 0.6704。
- ProfileLogit: F1 0.6828, AUC 0.5301。

Visual: 条形图，突出 F1 和 AUC。

## 8. Capacity Expansion

Title: Larger Prediction Heads Improve, but Do Not Explain Everything

Key message:
- 将 Logistic head 扩展为 MLP / DeepMLP 后性能提升。
- 最强 DeepMLP-2048x1024 达到 F1 0.7554, AUC 0.7954。
- 但参数量扩张后还需要消融检查各通道是否真实有效。

Table:
- DeepMLP-2048x1024: F1 0.7554, AUC 0.7954, params 2.61M。
- BoundaryGNN-Logit: F1 0.7367, AUC 0.7600, params 248。

Visual: 模型容量 vs F1 折线或柱状图。

## 9. Ablation Insight

Title: Ablation Reveals Channel Redundancy

Key message:
- 去掉 All-Neighbor Mean 后 F1 明显下降，说明普通邻居上下文重要。
- 去掉 Structure 后 AUC 下降明显，说明结构位置重要。
- 去掉 Similar 后 F1 略升，说明相似邻居通道在大模型下可能冗余。
- 这推动我们从固定拼接转向可学习通道选择。

Table:
- Full DeepMLP: F1 0.7554, AUC 0.7953。
- w/o All: F1 0.7449, AUC 0.7811。
- w/o Similar: F1 0.7581, AUC 0.7951。
- w/o Structure: F1 0.7461, AUC 0.7747。

Visual: 消融条形图，标注 w/o Similar 的反常现象。

## 10. Global Channel Gate

Title: Global Gate: Learn Which Channels to Trust

Key message:
- 为五个通道学习全局权重。
- 它能缓解固定拼接导致的通道冗余。
- 5 seed 下 F1 提升到 0.7597。

Gate means:
- self 1.002。
- all 1.003。
- similar 0.997。
- dissimilar 0.998。
- structure 0.998。

Limitation:
- 全局 gate 只能给所有节点同一组权重，表达能力仍然有限。

Visual: 五个通道旁边有一组全局 alpha。

## 11. Node-Adaptive Gate

Title: Node-Adaptive Gate: Different Nodes Use Different Channels

Key message:
- 每个节点根据自身五通道表示生成自己的 gate。
- 初始 gate 等于 1，训练后学习节点级通道选择。
- 解决“某些节点需要 similar 通道，某些节点不需要”的异质性。

Formula:
alpha_v = GateMLP(concat(self_v, all_v, similar_v, dissimilar_v, structure_v))

h_v = concat(alpha_v,self self_v, alpha_v,all all_v, alpha_v,similar similar_v, alpha_v,dissimilar dissimilar_v, alpha_v,structure structure_v)

Visual: 五通道输入 GateMLP，输出 per-node alpha_v，再进入 DeepMLP classifier。

## 12. Final Multi-Seed Result

Title: NodeGated BoundaryGNN Achieves the Best Mean F1 and AUC

Key message:
- NodeGated 在 5 seed 下取得最高 F1 和 AUC。
- 相比 global gate，F1 从 0.7597 提升到 0.7618，AUC 从 0.7923 提升到 0.7955。
- 它更适合边界风险节点的排序与筛查。

Table:
- NodeGatedDeepMLP: F1 0.7618 ± 0.0035, AUC 0.7955 ± 0.0035。
- Global GatedDeepMLP: F1 0.7597 ± 0.0007, AUC 0.7923 ± 0.0053。
- DeepMLP: F1 0.7568 ± 0.0030, AUC 0.7921 ± 0.0042。
- BoundaryGNN-Logit: F1 0.7363 ± 0.0060, AUC 0.7579 ± 0.0060。

Visual: 5 seed mean ± std bar chart。

## 13. Gate Interpretation

Title: What Did NodeGated Learn?

Key message:
- Global gate 基本接近 1，只做轻微校准。
- NodeGated 明显压低 similar 和 dissimilar 通道，同时保留 all-neighbor 通道。
- 这解释了消融中 similar 通道冗余的问题：不是完全无用，而是需要节点级选择。

Gate means:
- NodeGated: self 0.944, all 1.017, similar 0.895, dissimilar 0.919, structure 0.937。
- Global: self 1.002, all 1.003, similar 0.997, dissimilar 0.998, structure 0.998。

Visual: gate heatmap 或 paired bar chart。

## 14. Negative Result: Soft Labels

Title: Soft Labels Were Tried but Not Kept

Key message:
- Soft-label training 将 hard label 转为连续风险分数。
- 实验显示 recall 略高，但 precision、F1、AUC 下降。
- 当前任务更接近明确的边界节点二分类，soft target 会磨钝决策边界。

Result:
- Gated hard: F1 0.7596, AUC 0.7943。
- Gated soft: F1 0.7527, AUC 0.7754。

Visual: 简单对比表，作为实验完整性展示。

## 15. Contributions

Title: Contributions

Key message:
1. 将 SNAP Google+ Circles 重构为社交边界风险节点识别任务。
2. 提出基于画像相似度边分型的 BoundaryGNN 表示。
3. 通过消融发现固定多通道拼接存在冗余。
4. 提出 Node-Adaptive Gate，实现节点级通道选择。
5. 在 122 个 ego networks、39,204 labeled nodes 上完成 baselines、容量分析、消融、soft-label 探索和 5 seed 稳定性实验。

Visual: 五个贡献点横向流程。

## 16. Conclusion

Title: Conclusion

Key message:
- 社交边界风险识别需要同时建模画像、结构和跨圈层邻居关系。
- 简单堆参数有效但不够，消融显示通道选择才是关键。
- NodeGated BoundaryGNN 在多随机种子下取得最佳 F1 和 AUC，是当前最终方法。

Final line:
From fixed aggregation to node-adaptive channel selection.

Visual: 最终方法图 + 最终结果小表。
