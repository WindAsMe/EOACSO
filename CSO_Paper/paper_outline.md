# 论文提纲（生物医学工程/信号处理类期刊定位）

> 目标期刊类型：Biomedical Signal Processing and Control / Computers in Biology and Medicine 一类
> 定位调整：这类期刊接受完整的算法技术细节，但读者以生物医学工程/临床信息学背景为主，
> 论文的"卖点"应该是**一个可用于PD语音筛查的诊断流程**，算法创新是支撑这个流程的手段，
> 而不是反过来把算法本身当作唯一卖点。相应地：临床背景、生物标志物的生理解释、
> 筛查工具的临床定位与局限性，都要比纯CS版本的提纲写得更重、更具体。

## 0. 题目 / Title
- **"A Fully Searched Voice Analysis Pipeline for Parkinson's Disease Screening: Joint Feature, Classifier, and Transfer-Function Optimization via Competitive Swarm Optimization"**
  （筛查流程为主语，方法为副标题；现在的方法是普通 CSO——不是EOACSO——"创新"落在联合搜索的编码本身，尤其是把 transfer function 的选择也纳入搜索。注：正文全文的表述顺序已统一改为"Transfer Function-, Feature-, Classifier-"，题目副标题的顺序未来若统一调整需一并检查）

## 1. Abstract（建议用结构化摘要：Background / Objective / Methods / Results / Conclusion）
- **Background**：PD患病率与早期诊断的临床价值；~90%患者有言语障碍(dysphonia)，且可能早于/伴随运动症状出现，是潜在的低成本、非侵入性生物标志物
- **Objective**：现有基于声学特征+机器学习的PD筛查研究，普遍存在三个问题：(a) 特征维度高但很少同时优化"用哪个分类器"；(b) 特征选择与分类器选择即使联合优化也多为两阶段串行，而非单一搜索过程内联合决策；(c) 二值化搜索空间所用的transfer function本身几乎总是外部固定选定，而非纳入搜索——本文目标是提出一个联合搜索transfer function、声学特征子集与分类器配置的统一筛查流程
- **Methods**：普通CSO（Competitive Swarm Optimizer）+ 每个粒子独立搜索自己的transfer function（5个候选：2个S-shaped、2个V-shaped、1个hard threshold，argmax解码）；联合编码 $5+D+5$（selector + features + classifiers）；两个公开声学数据集；分组交叉验证防止受试者信息泄漏；与7个近年发表的同类算法（复现自5篇源论文）在统一协议下对比；CSO社会项系数 $\phi$ 的敏感性分析
- **Results**：（跑完实验后填，用敏感度/特异度/AUC等临床更熟悉的指标做主语）
- **Conclusion**：一句话概括方法的诊断性能 + 明确定位为"筛查辅助工具"，非诊断替代

## 2. Introduction
1. **PD的临床负担与早期诊断意义**（独立成段，篇幅要够）
   - 患病率、老龄化背景下的增长趋势
   - 现行诊断高度依赖临床观察运动症状（MDS临床诊断标准）+左旋多巴治疗反应，主观性强、早期误诊率高
   - 早期干预对延缓病程/提高生活质量的临床价值（呼应现有文献）
2. **言语障碍作为PD生物标志物的生理基础**（这是医学期刊读者最关心的一段，务必写扎实）
   - 喉部肌肉强直/运动迟缓 → 基频微扰(jitter)、振幅微扰(shimmer)、谐噪比(HNR)等声学指标异常的生理机制
   - 非运动症状可能早于运动症状出现，语音改变可能是早期可捕捉的信号
   - 非侵入、低成本、可远程采集（呼应远程医疗/居家筛查场景）
3. **现有基于语音的PD机器学习检测研究综述**（简要）
   - 已取得的准确率水平，几类主流做法（滤波式/包裹式特征选择、集成学习、深度学习）
4. **现有工作的方法学问题**（技术贡献的铺垫，但要服务于"为什么这对临床应用重要"）——已在正文实现为三点，措辞顺序统一为"transfer function → feature → classifier"：
   - (a) 二值化搜索所用的transfer function几乎总是外部固定选定（S-shaped/V-shaped等惯例），比较研究表明没有单一选择能全面主导，这本身也该是搜索决定的对象，而非事先拍板（引用 Kuntalp:2024）
   - (b) 特征选择很少联合分类器选择一起优化，即使联合优化，分类器类型本身通常也固定，只调超参数 → 影响：选出的特征子集可能不是最优诊断组合
   - (c) 跨研究不可比：各自用各自的分类器/数据切分/是否按受试者分组CV（很多研究存在同一受试者的重复录音同时进入训练和测试集的数据泄漏风险，导致报告的准确率虚高，临床意义存疑）→ 这一点要明确点名批评并作为本文方法学卖点
5. **本文贡献**（bullet list，措辞偏向"诊断工具"而非"新算法"，与introduction.tex现有4条一致）：
   - 提出在单一优化编码内，联合搜索"用哪个transfer function做二值化"、声学特征子集、分类器配置，而非只做特征选择或事先固定二值化方案
   - 在普通CSO之上扩展出per-particle的transfer-function-selector编码段，使种群中每个粒子可独立搜索5个候选二值化方案中的一个，而非全种群共享一个预先固定的方案
   - 在统一编码、适应度函数与评估协议下，与7种近期发表的元启发式特征选择方法系统比较，验证竞争力
   - 结合已知的PD构音障碍生理机制，考察本文流程最常选中的声学特征，评估结果的生物学合理性与临床可解释性
6. 段落结构导览

## 3. Related Work
### 3.1 帕金森病的诊断现状与挑战
- 临床诊断标准、误诊率、对客观生物标志物的临床需求
### 3.2 言语声学生物标志物与PD的关联
- 详细一点：哪些声学特征已被证实与PD相关、机制是什么（这部分内容可以从 Oxford 数据集文档 + 已有临床文献整理）
### 3.3 基于语音/机器学习的PD检测研究综述
- 按"是否分组CV/是否联合优化分类器/是否搜索transfer function/复现难度"这几个维度做对比表，直接为本文的方法学定位铺垫
### 3.4 特征选择元启发式方法（技术性内容，适度精简）
- 群体智能优化在生物医学特征选择中的应用；CSO简介（大规模优化背景、竞争式更新 vs. PSO式吸引子更新，见related_work.tex已有段落）
- 收尾呼应：transfer function的选择在文献中几乎总是外部固定，为本文"把它也纳入搜索"的动机铺垫
### 3.5 本文对比的基线方法
- 建议用一张**表格**（算法名/来源/年份/核心思路/原文报告的数据集与准确率）代替逐篇展开小节：
  实际复现8个（5篇源论文：Hashemi:2026, AlNajjar:2024, Santhosh:2025, Hashim:2023, Mansour:2024），其中BWOA因结果异常、与源论文结论不一致被排除，报告对比的基线为**7个**（BPSO, BGWO, HybridGWO, MGWO-eP, mHGS, QMFO，以及被排除的BWOA需在正文/讨论中说明排除理由）

## 4. Materials and Methods
> 医学期刊习惯用 "Materials and Methods"，且顺序通常是先数据、后方法（跟CS论文"先方法后数据"相反）
### 4.1 数据集与研究对象
- Oxford数据集：受试者人口学信息（性别/年龄段/PD病程如果数据里有）、录音协议、样本量
- Naranjo数据集：同上
- 强调：均为已发表、公开可及的回顾性数据集，非本研究自行招募
### 4.2 数据预处理与分组交叉验证协议
- 按subject分组的5折CV，明确说明这是为了避免同一受试者的重复录音跨fold造成的数据泄漏
  （这是全文方法学严谨性的一个核心卖点，值得单独成节而不是塞进方法细节里）
### 4.3 Transfer-Function-, Feature-, and Classifier- 联合优化编码
- 基线：$D+5$维编码（特征子集 + 分类器选择），用各自源论文固定的transfer function二值化
- 本文提案：在此基础上前置一个5维transfer-function-selector段，总长 $5+D+5$，布局为 $[t_1,\dots,t_5 \mid f_1,\dots,f_D \mid c_1,\dots,c_5]$；selector段通过argmax解码（不做二值化），决定后面 $D+5$ 维具体用哪个候选transfer function二值化
- 候选分类器（SVM/RF/XGBoost/kNN/LogReg）及各自的临床/工程适用场景简述；分类器编码支持multi_hot（默认，等权软投票集成）与top1（对齐各基线原文的单分类器方案）两种，全部算法在同一协议下可切换比较
### 4.4 候选Transfer Function（二值化方案）
- 5个候选：classic S-shaped、Hashemi's S-shaped（方向相反的sigmoid）、erf-based V-shaped、tanh-based V-shaped（Mirjalili & Lewis V2）、hard threshold
- 每个基线固定使用其源论文指定的方案；本文提案的5个候选构成search space的一部分，由selector段决定，而非事先固定
### 4.5 适应度函数
- 平衡准确率 + 特征简约惩罚，公式与动机（简约性对应"用尽量少的声学测量做筛查"的临床实用性）
### 4.6 本文提出的流程：基于CSO的联合优化（Our Proposed Pipeline）
- 明确措辞：贡献不是新的元启发式算子，而是"给已有的CSO一个新的搜索对象"——联合编码 + transfer-function-selector解码步骤，而非CSO更新规则本身的修改（避免让读者误读为"提案=普通CSO，没有新东西"）
- CSO基础机制回顾：竞争式配对、winner不变/loser更新、社会项系数 $\phi$（Cheng & Jin 2015 Eq. 25-26），本研究种群规模下 $\phi=0$
- Per-particle transfer-function selection的解码流程（argmax selector → 二值化 trailing $D+5$ → 评估）
### 4.7 对比方法
- 7个复现基线（引用3.5的表）+ 排除BWOA的说明；不再包含消融研究（已确认无必要，见下方"变更记录"）
- 统一评估预算（max_evaluations）说明
### 4.8 评价指标
- Accuracy, Balanced Accuracy, **Sensitivity(Recall)**, **Specificity**, Precision, F1, ROC-AUC —— 医学期刊里 Sensitivity/Specificity 通常要放在比 F1/ROC-AUC 更突出的位置，因为这是临床筛查工具评估的标准语言
- 特征选择率（对应"筛查所需声学测量数量"这一实用性指标）
### 4.9 统计分析
- Wilcoxon signed-rank / Friedman test，配对设计说明（同run_idx跨算法共享随机种子/CV划分）
### 4.10 实现细节
- 软件环境、计算资源、独立运行次数

## 5. Results
### 5.1 研究队列描述性统计
- 两个数据集的样本特征表（Table 1 风格，医学论文标配）
### 5.2 诊断性能对比：提案方法 vs. 复现基线
- 主表：Accuracy/Sensitivity/Specificity/Precision/F1/AUC，mean±std，两个数据集分别列
- 收敛曲线图
### 5.3 统计显著性
- Wilcoxon/Friedman结果
### 5.4 CSO社会项系数 φ 的敏感性分析（新增）
- 固定 $\phi \in \{0, 0.05, 0.10, \dots, 0.50\}$（11个取值）重跑联合优化，观察诊断性能/收敛行为随 $\phi$ 的变化
- 说明本研究实际种群规模下默认 $\phi=0$（原论文公式在 $N\le100$ 时恒为0），这组实验是刻意偏离默认值做的敏感性检验，而非复现原论文行为
- 数据来源：`src/experiments/run_phi_sensitivity.py` → `results/tables/*_phi_sensitivity_results.csv`
### 5.5 筛查所需声学特征分析（核心的"临床可解释性"章节）
- 高频被选中的特征列表 + 对应的生理意义（回扣3.2节的机制介绍）
- 讨论这些特征是否与已发表的PD声学标志物文献吻合，作为"结果生物学合理性"的佐证
- （如果有）精度-简约度权衡的帕累托解集，讨论"用更少特征也能维持可接受敏感度/特异度"对居家/远程筛查场景的意义
### 5.6 分类器与transfer function选择偏好分析
- 不同数据集/算法下最常被选中的分类器，简短讨论
- 本文提案在不同数据集/运行间最常被argmax选中的transfer function，简短讨论（对应"文献里没有单一transfer function全面主导"这一动机的实证呼应）

## 6. Discussion
### 6.1 主要发现总结
### 6.2 与既有文献的比较（定性）
- 强调协议差异（分组CV vs 非分组），解释为何不能直接比较绝对数字，但可以比较相对趋势
### 6.3 临床意义与潜在应用场景
- 远程/居家筛查、初级医疗转诊分流工具、随访监测（非诊断替代，需明确措辞）
- 对"简约特征子集"在实际部署（便携设备/App采集少量声学指标）上的意义单独讨论
### 6.4 局限性（这一节要写得比纯CS版本更"诚实"和具体，医学审稿人非常看重这部分）
- **回顾性、小样本、单一/双数据集**：外部有效性有限，未做多中心/前瞻性验证
- **人群多样性有限**：年龄/性别/病程分布未知或不均衡（如数据集本身有这个问题要如实说）
- **录音条件标准化程度未知**：原始数据集文档缺失采样率等细节
- **无法与临床医生诊断准确率做直接对照**：本文报告的是"相对于金标准标签"的分类性能，非"相对于临床专家判断"的一致性
- **transfer function候选池本身是有限、预先设定的5种**：搜索的是"从这5种里选哪个"，不是二值化方案本身的开放式设计，需说明这一范围限定
- **复现基线时的方法学透明度**：各基线论文部分公式/参数缺失，本文做了文档化的合理假设（可以简要提，细节放Supplementary）
- **BWOA排除**：需说明排除理由（结果异常、与源论文结论不一致），并讨论这对"8选7"基线覆盖面的影响
### 6.5 未来工作
- 前瞻性、多中心队列验证；与临床评分（如UPDRS）的相关性分析；纵向随访设计；实时/移动端部署可行性研究；扩大transfer function候选池或允许候选池本身参数化

## 7. Conclusion
- 一段话：方法+性能+明确的临床定位（辅助筛查工具，需要前瞻性验证）

## 8. References

## 9. Supplementary Material（建议）
- 提案流程（CSO + per-particle transfer-function selection）完整伪代码与超参数
- 7个复现基线的公式复现细节与"gap-fill"透明度说明（对应各`src/optimizers/*.py` docstring），含BWOA的排除说明
- 完整的φ敏感性分析数据表

---

## 变更记录（相对更早版本提纲的重大调整）
- 方法从 EOACSO（CSO + 精英引导更新 + OBL + 多样性精英档案）改回**普通、未修改的CSO**；新贡献改为把transfer function的选择也纳入搜索（per-particle、5候选、argmax解码）
- 编码从 $D+5$（baseline写法）扩展为提案方法的 $5+D+5$（selector + features + classifiers），正文统一用未简化的 $5+D+5$ 写法
- **消融研究已删除**（不再需要，5.4/9节的"消融"内容已移除）
- 全文三项联合优化的表述顺序统一改为 **"Transfer Function-, Feature-, Classifier-"**（不是Feature-, Classifier-, Transfer-Function-）
- 新增 5.4 节：CSO社会项系数 $\phi$ 的敏感性分析（$\phi \in \{0, 0.05, \dots, 0.50\}$，11个取值），对应新脚本 `src/experiments/run_phi_sensitivity.py`
- 基线从"8个"改为"7个对比+1个排除（BWOA）"的表述

## 附：写作时可直接复用的项目产出物
- `results/tables/*_comparison_results.csv` —— 5.2 主表数据来源
- `results/tables/*_phi_sensitivity_results.csv` —— 5.4 敏感性分析数据来源（`src/experiments/run_phi_sensitivity.py`，跑完后才有数据）
- `history` 字段（JSON数组）—— 5.2 收敛曲线图
- `src/experiments/stats.py` 的 Wilcoxon/Friedman —— 5.3 数据来源
- `active_classifiers`/`feature_mask`/`transfer_function` 相关字段 —— 5.5/5.6 特征、分类器、transfer function偏好分析的数据来源
- 各 `src/optimizers/*.py` 的 docstring —— Supplementary Material 的 gap-fill 说明素材
- `src/optimizers/cso.py`、`src/optimizers/transfer.py` —— 4.3/4.4/4.6 方法描述草稿（编码、候选transfer function、CSO本体）
- **需要你补充的医学文献素材**：PD诊断标准(MDS Criteria)引用、语音生理机制相关综述、现有PD语音ML研究的对比文献——这几块目前项目代码/README里没有现成素材，需要你从医学文献库另外检索补充
