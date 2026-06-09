# OpenBioMed 平台模块与接口调研

> 目标：按照“大模型平台架构 v0.1”的模块划分，调研 OpenBioMed 仓库中是否存在对应模块，并整理模块之间的箭头/接口关系，包括输入、输出数据类型、工具调用与缺口。

## 1. 总体结论

OpenBioMed 不是一个完整的“任务队列 + 多执行器 + 算法评测器”生产平台实现，而是一个以 **OpenBioMed 数据对象 + Tool 注册表 + Pipeline/Task/Model 推理 + Workflow DAG + Agent PlannerExecutor + FastAPI 服务** 为核心的生物医学大模型工具平台。

与参考架构对齐后可以得到：

| 参考架构模块 | OpenBioMed 对应实现 | 存在程度 | 说明 |
|---|---|---:|---|
| 应用层 | `open_biomed/scripts/run_server.py`、`run_server_workflow.py`、`open_biomed/scripts/chat.py`、`examples/`、`demo_workflows/` | 部分存在 | 后端 API、脚本、示例存在；仓库内没有完整前端应用源码。 |
| 任务规划器 / 智能路由 | `open_biomed/core/agent.py::PlannerExecutor`、`TOOLS`/`WORKFLOWS` 注入 prompt、`parse_frontend()` | 存在 | LLM 根据工具说明规划并执行；前端导出的图可转 YAML workflow。 |
| 任务队列管理器 | `Workflow.exec_queue`、LangGraph state、`asyncio.create_task()` | 弱存在 | 有内存队列/执行队列，不是持久化任务队列，也没有优先级、重试队列、资源调度队列。 |
| 任务调度器 / 管线隔离 | `Workflow` DAG 拓扑执行、`InferencePipeline`、可选 Docker 执行、可视化 subprocess | 部分存在 | Workflow 是 DAG 调度；可视化用子进程规避 PyMol 问题；Agent 可选 Docker，但没有统一集群调度/隔离层。 |
| 内部数据 | `open_biomed/data/*`、`datasets/*`、`memory/workflows/*`、`configs/*` | 存在 | 统一对象包括 Molecule、Protein、Pocket、Text、Cell；训练/评测数据集也有 Registry。 |
| 外部数据 | `web_request_tools.py`：PubChem、UniProt、PDB、STRING、ChEMBL、WebSearch 等 | 存在 | 通过异步 HTTP 查询外部数据库/搜索服务，返回 OpenBioMed 对象或结构化数据。 |
| 靶点发现执行器/算法仓库 | `protein_binding_site_prediction`、`protein_question_answering`、`ppi_string_request`、skills 中靶点相关能力 | 部分存在 | 代码层主要是结合位点/PPI/蛋白问答；没有独立命名的“靶点发现算法仓库”。 |
| 分子生成执行器/算法仓库 | `structure_based_drug_design`、`text_guided_molecule_generation`、`text_based_molecule_editing`、`pocket_molecule_docking` | 存在 | 支持基于口袋生成、文本引导生成/编辑、对接。 |
| 性能预测执行器/算法仓库 | `molecule_property_prediction`、`molecule_qed/sa/logp/lipinski/similarity`、`protein_molecule_docking_score` | 存在 | ML 预测 + 规则/化学指标 + docking score。 |
| 合成规划执行器/算法仓库 | skills 中 `retrosynthesis-planning`，但核心 `TOOLS` 无对应工具 | 弱存在 | Skills 层有逆合成规划；核心 Python tool registry 没有合成路线规划接口。 |
| 临床优化执行器/算法仓库 | `molecule_property_prediction`、`text_based_molecule_editing`、skills 中 ADMET/lead analysis | 部分存在 | 更接近 ADMET/性质优化；没有临床试验/真实世界证据优化执行器。 |
| 算法评测器 | `TrainValPipeline`、Task callbacks/monitor、`protein_molecule_docking_score`、性质计算工具 | 部分存在 | 有训练/测试评测和若干任务指标工具，但没有统一“新算法接入-评测-发布”平台模块。 |

## 2. 核心数据接口

OpenBioMed 的模块间箭头主要不是传裸 JSON，而是传 Python 对象；HTTP 层再把文件路径、SMILES/FASTA、文本转换为对象。

### 2.1 数据对象

| 数据类型 | 构造输入 | 主要字段 | 输出/序列化 | 典型流向 |
|---|---|---|---|---|
| `Molecule` | SMILES、SELFIES、RDKit Mol、PDB/SDF/PDBQT/PKL 文件 | `smiles`、`rdmol`、`graph`、`conformer`、`description`、`kg_accession` | `.sdf`、`.pkl`、字符串 SMILES | 分子问答、性质预测、生成/编辑、对接、可视化、导出 |
| `Protein` | FASTA、PDB/PKL 文件 | `sequence`、`residues`、`all_atom`、`description`、`kg_accession` | `.pdb`、`.pkl`、字符串序列 | 蛋白问答、折叠、突变设计、结合位点预测、可视化 |
| `Pocket` | 蛋白子序列 residue indices、蛋白+参考配体、PDB/PKL 文件 | `atoms`、`conformer`、`orig_indices`、`orig_protein` | `.pdb`、`.pkl` | SBDD、对接、口袋可视化 |
| `Text` | 字符串 | `str` | 字符串 | QA、编辑/优化 prompt、摘要 |
| `Cell` | `scanpy.AnnData` 或序列 | `anndata`、`sequence` | 对象 | 单细胞注释任务 |

### 2.2 统一 Tool 输入转换

`create_tool_input(data_type, value)` 的约定：

| `data_type` | 输入字符串解释 | 输出对象 |
|---|---|---|
| `molecule` | `.sdf` -> `Molecule.from_sdf_file`；`.pkl` -> `Molecule.from_binary_file`；否则按 SMILES | `Molecule` |
| `protein` | `.pdb` -> `Protein.from_pdb_file`；同名 `.pdb` 存在时优先；`.pkl` -> `Protein.from_binary_file`；否则按 FASTA | `Protein` |
| `pocket` | `.pkl` | `Pocket` |
| `text` | 原字符串 | `Text` |
| 其他 | 原值 | `Any` |

### 2.3 Tool 标准返回

所有 Tool 遵循 `run(...) -> Tuple[List[Any], List[Any]]`：

- 第一个列表：真实输出对象，供后续工具传递。
- 第二个列表：观测/文件路径/前端展示文本，供 UI、日志或下载使用。
- `serial_exec` 装饰器支持把 list 输入逐条执行并拼接输出。
- `wrap_outputs()` / `wrap_and_select_outputs()` 根据输出对象类型封装成 `molecule/protein/pocket/text/output`，供 workflow 边自动注入下游输入。

## 3. 模块级接口与箭头

### 3.1 应用层 -> 任务规划器/工具服务

#### HTTP API

| Endpoint | 请求模型 | 关键输入字段 | 调用目标 | 返回 |
|---|---|---|---|---|
| `POST /run_pipeline/` | `TaskRequest` | `task`, `model`, `molecule`, `protein`, `pocket`, `text`, `dataset`, `mutation`, `indices`, `property` 等 | `TASK_CONFIGS` -> `TOOLS[pipeline_key]` -> handler | 任务名 + 模型 + molecule/protein/pocket/text/score/image/file 等 |
| `POST /web_search/` | `SearchRequest` | `task`, `query`, `molecule`, `threshold` | 异步 requester tool | 外部数据查询结果，例如 molecule/protein/text |
| `POST /run_workflow/` | `ReportRequest` | `task`, `workflow`, `user_email`, `num_repeats` | `ReportGeneratorSBDD` 或 `ReportGeneratorGeneral` | 立即返回“Workflow is still running...”，后台异步生成报告 |

#### 脚本/Chat

- `open_biomed/scripts/inference.py` 提供一组 `test_*` 构造函数，把下游任务封装为 `InferencePipeline` 或 `VinaDockTask`，并被 `TOOLS` 懒加载。
- `open_biomed/core/agent.py::PlannerExecutor` 面向自然语言任务：先生成计划，再在 `<execute>` 中执行 Python/Bash，或在 `<report>` 中输出报告。

### 3.2 任务规划器 / 智能路由

#### 实现

- `PlannerExecutor` 初始化：读取 agent 配置，创建 LLM，选择 context manager，初始化工具、memory workflow、system prompt、执行环境和 LangGraph 工作流。
- 工具注入：若未配置 tool retriever，则把 `TOOLS.available_tools()` 中所有工具的 `print_usage()` 拼到 prompt。
- Workflow memory：若开启 `memory.workflow`，把 `WORKFLOWS` 描述、输入、输出注入 prompt，允许 Agent 直接调用预定义 workflow。
- 执行语言：Python 或 Bash；Python 复用持久 namespace，Bash 可选 Docker。

#### 输入

| 输入来源 | 数据结构 |
|---|---|
| 用户自然语言 | `user_prompt: str` |
| Agent 配置 | `timeout`, `plan_style`, `critic`, `tool_call`, `use_docker`, `memory`, `llm`, `context_manager` 等 |
| 工具说明 | `TOOLS[tool].print_usage()` |
| Workflow 说明 | `workflow.metadata` |

#### 输出

| 输出 | 说明 |
|---|---|
| `thread_id` | 会话目录 `tmp/planner_executor-{thread_id}` |
| `messages` | LangGraph/对话上下文 |
| `captured_results` | 保存的 figure/molecule/protein/visualization 文件 |
| `report.md/pdf` | 若 LLM 输出 `<report>`，可导出报告 |

#### 路由逻辑

1. LLM 生成包含 `<execute>` 或 `<report>` 的响应。
2. 若是 `<execute>`，执行 Python/Bash，并把 stdout/stderr 作为 observation 回写。
3. 若是 `<report>`，结束。
4. 连续解析失败或代码失败达到容忍阈值后结束或要求重试。

### 3.3 前端 Workflow 图 -> YAML Workflow

`parse_frontend(json_string)` 把前端节点/边 JSON 转换为内部 YAML：

- 读取 `nodes` 与 `edges`。
- 过滤 `ChatInput`、`ChatOutput`、`ParseData` 等前端节点。
- 提取 tool 节点的 `config/molecule/protein/pocket/text/dataset/query/mutation/indices/threshold` 等输入。
- 合并 `MergeDataComponent`、`ParseData` 边。
- 生成：
  - `tools: [{name, inputs?}, ...]`
  - `edges: [{start, end, name_mapping?}, ...]`
- 对若干前端字段做硬编码映射，例如 `molecule_property_prediction.dataset -> task`、`molecule_name_request.query -> accession`。

### 3.4 任务队列管理器与任务调度器

OpenBioMed 的调度是 **DAG 内存调度**：

| 模块 | 实现 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| Workflow DAG 构建 | `Workflow(config)` | YAML `metadata/tools/edges` | `nodes`, `edges`, `output_nodes` | tool name 解析到 `TOOLS`；`code_execution` 特殊处理 |
| 执行队列 | `exec_queue`, `in_deg` | DAG 入度 | 待执行节点序号 | 按拓扑顺序执行；无持久化/优先级 |
| 边传递 | `wrap_outputs()` + `name_mapping` | 上游 Tool 输出对象列表 | 下游 node.inputs | 默认按类型名注入；可通过 `name_mapping` 改字段名 |
| 重复执行 | `num_repeats` | 节点配置 | 多次输出 append | 节点级重复 |
| 中断 | `deamon.should_interrupt(state)` | runtime state | bool | 可中断，但没有完整外部调度器 |

箭头形式：

```text
Application/API/Agent
  -> Config/YAML Workflow
  -> Workflow(nodes, edges)
  -> exec_queue 按拓扑取 node
  -> TOOLS[node.name].run(**inputs)
  -> wrap_outputs(outputs)
  -> edge.name_mapping 重命名
  -> downstream_node.inputs
  -> output_nodes results/messages
```

### 3.5 数据层：内部数据与外部数据

#### 内部数据

- `open_biomed/data/`：运行时数据对象。
- `open_biomed/datasets/`：训练/评测数据集；通过 `DATASET_REGISTRY` 被 `DefaultDataModule` 载入。
- `configs/`：模型、数据集、可视化、workflow 配置。
- `memory/workflows/`：Agent 可用的预定义 workflow memory。
- `skills/`：更高层的生物医学工作流能力说明与脚本资产。

#### 外部数据工具

| Tool | 外部系统 | 输入 | 输出 |
|---|---|---|---|
| `molecule_name_request` / `pubchemid_search` | PubChem compound by CID/name | `accession`/`query` | `Molecule`, `.pkl` 路径 |
| `molecule_structure_request` | PubChem similarity API | `Molecule`, `threshold`, `max_records` | 相似 `Molecule` 列表 |
| `pubchem_bioactivity` | PubChem PUG View / bioactivity | molecule/name/id | bioactivity dict |
| `protein_uniprot_request` | UniProt | accession | `Protein` 或 FASTA/PKL 路径 |
| `protein_pdb_request` | PDB / AlphaFoldDB | accession, mode | `Protein`、PDB 文件路径或 metadata |
| `ppi_string_request` | STRING | UniProt/gene, species, score, limit | interaction partner dict |
| `chembl_query` | ChEMBL | molecule/target/indication query | activity/phase dict |
| `web_search` | 搜索引擎 | query | 拼接文本结果 |

### 3.6 执行器与算法仓库

OpenBioMed 通过 `TOOLS` 将执行器暴露为统一工具；重模型工具一般是 `InferencePipeline`，轻量/外部工具是 `Tool` 子类。

#### 3.6.1 靶点发现相关

| Tool/能力 | 输入 | 输出 | 底层 |
|---|---|---|---|
| `protein_binding_site_prediction` | `protein: Protein` | `Pocket` 列表 + `.pkl` 路径 | P2Rank CLI (`third_party/p2rank_2.5/prank`) |
| `protein_question_answering` | `protein: Protein`, `text: Text` | `Text` answer | BioT5 pipeline |
| `ppi_string_request` | `uniprot_id`, `species`, `required_score`, `limit` | PPI partners/confidence | STRING API |
| `protein_pdb_request` / `protein_uniprot_request` | accession | Protein/PDB/metadata | PDB/UniProt API |

#### 3.6.2 分子生成 / 设计相关

| Tool/能力 | 输入 | 输出 | 底层模型/算法 |
|---|---|---|---|
| `structure_based_drug_design` | `pocket: Pocket` | `Molecule` with likely pocket binding | MolCRAFT or PharmolixFM checkpoint configuration |
| `text_guided_molecule_generation` | `text: Text` | `Molecule` | task/model registry 支持，未在 server `TOOLS.available_tools()` 中暴露 |
| `text_based_molecule_editing` | `molecule: Molecule`, `text: Text` | edited `Molecule` | MolT5/BioT5 style pipeline |
| `pocket_molecule_docking` | `molecule: Molecule`, `pocket: Pocket` | `Molecule` with 3D binding pose | PharmolixFM pipeline |
| `mutation_engineering` | `protein: Protein`, `text: Text` | mutation strings；server 进一步转 mutated `Protein` | MutaPLM pipeline |
| `go_guided_protein_generation` | GO term list | protein sequence | CodeFP pipeline |

#### 3.6.3 性能预测 / 打分相关

| Tool/能力 | 输入 | 输出 | 底层 |
|---|---|---|---|
| `molecule_property_prediction` | `molecule: Molecule`, `task/dataset` | property score/text | GraphMVP ensemble, BBBP/SIDER/regression 等 |
| `molecule_qed` | `molecule: Molecule` | QED float | RDKit QED |
| `molecule_sa` | `molecule: Molecule` | SA score | RDKit + fpscores |
| `molecule_logp` | `molecule: Molecule` | LogP float | RDKit Crippen |
| `molecule_lipinski` | `molecule: Molecule` | Lipinski count | RDKit descriptors |
| `molecule_similarity` | `molecule_1`, `molecule_2` | fingerprint similarity | RDKit fingerprint |
| `protein_molecule_docking_score` | `molecule: Molecule`, `protein: Protein` | Vina score / pose | VinaDockTask |

#### 3.6.4 合成规划相关

- 核心 Python `TOOLS` 中没有 `retrosynthesis_planning` / synthesis route planning tool。
- `skills/retrosynthesis-planning` 表明仓库在 Skills 层规划了该能力，但不属于当前 core/server 的可直接调用工具。
- 因此“合成规划执行器/算法仓库”在 core 平台中是缺口，在 skills 层是弱存在。

#### 3.6.5 临床优化相关

- 无独立“临床优化执行器”。
- 可组合能力包括：`molecule_property_prediction`（如 SIDER、BBBP、ADMET 类任务）、`text_based_molecule_editing`（文本目标优化）、`protein_molecule_docking_score`（结合评分）、skills 中 `admet-prediction`、`drug-lead-analysis`、`target-drug-report`。
- 更准确的定位：OpenBioMed 当前支持“先导化合物性质优化/ADMET 风险评估”，不支持临床流程调度或临床数据闭环优化。

### 3.7 算法评测器

OpenBioMed 的评测分散在三层：

| 层 | 实现 | 输入 | 输出 | 适用范围 |
|---|---|---|---|---|
| 训练/验证评测 | `TrainValPipeline.run()` | task + dataset + model config | Lightning test metrics/logs/checkpoints | 模型训练与 benchmark |
| Task callbacks/monitor | `BaseTask.get_monitor_cfg()`、callbacks | validation/test predictions | metrics, saved outputs | 具体任务 |
| 工具级打分 | Vina score、QED/SA/LogP/Lipinski/similarity/property prediction | Molecule/Protein/Pocket | score/text | workflow 中即时评价 |

缺口：没有统一“新算法提交 -> 自动标准化输入输出 -> 多数据集评测 -> 指标排行榜 -> 发布到算法仓库”的模块。若要贴合参考图中的“新算法 -> 算法评测器 -> 算法仓库”，需要新增统一算法插件规范和评测流水线。

## 4. 典型模块间箭头实例

### 4.1 结构化药物设计 workflow

以 `configs/workflow/stable_drug_design.yaml` 为例：

```text
输入: protein PDB + reference pocket PKL
  -> visualize_protein_pocket(protein, pocket) -> image
  -> structure_based_drug_design(pocket) -> molecule_0
      -> protein_molecule_docking_score(protein, molecule_0) -> score_0
      -> text_based_molecule_editing(molecule_0, text="lower liver toxicity") -> molecule_1
      -> pocket_molecule_docking(pocket, molecule_1) -> molecule_1_pose
          -> protein_molecule_docking_score(protein, molecule_1_pose) -> score_1
          -> visualize_complex(protein, molecule_1_pose) -> image
      -> molecule_question_answering(molecule_0/molecule_1, property questions) -> text answers
      -> molecule_property_prediction(molecule_0/molecule_1, task=SIDER) -> toxicity/ADR score
```

接口要点：`structure_based_drug_design` 输出 `Molecule`，通过 `wrap_outputs()` 自动以 `molecule` 字段注入下游；`protein`/`pocket` 是节点固定输入或上游输出。

### 4.2 蛋白定向进化 workflow

`configs/workflow/directed_evolution.yaml` 表达：

```text
UniProt accession
  -> protein_uniprot_request(accession) -> Protein
      -> protein_question_answering(Protein, motif/domain question) -> Text
      -> protein_question_answering(Protein, function question) -> Text
      -> mutation_engineering(Protein, desired property text) -> mutation
      -> apply_mutation_to_sequence(Protein, mutation) -> mutated Protein
      -> mutation_engineering(mutated Protein, desired property text) -> mutation
      -> apply_mutation_to_sequence(...)
      -> protein_folding(mutated Protein) -> structured Protein
      -> visualize_protein(structured Protein) -> image
```

其中 mutation 工具输出默认字段是 `output`，workflow 边通过 `name_mapping: output -> mutation` 把它改名成 `apply_mutation_to_sequence` 所需输入字段。

### 4.3 PDB 查询 workflow memory

`memory/workflows/pdb_query.yaml` 表达：

```text
PDB ID
  -> protein_pdb_request(mode=metadata) -> metadata JSON/text
      -> code_execution(JSON dumps + collect) -> content list
      -> summarize_content(content) -> summary
  -> protein_pdb_request(mode=file_only) -> pdb_file
      -> extract_molecules_from_pdb_file(pdb_file) -> proteins/ligands/ions list
```

该 workflow 是 Agent memory，可被 `PlannerExecutor` 在自然语言任务中调用。

## 5. 与参考架构的差距与建议

### 5.1 已具备能力

- 统一 Tool 接口和懒加载工具注册表。
- 统一数据对象，支持对象/文件两种传递形式。
- 预定义 workflow DAG，支持 name mapping 和节点重复执行。
- Agent 能把自然语言、工具说明、workflow memory、代码执行环境连接起来。
- HTTP 服务把外部应用请求转成 Tool 调用。
- 外部数据库/搜索工具覆盖 PubChem、UniProt、PDB、STRING、ChEMBL。

### 5.2 主要缺口

1. **任务队列管理器不足**：当前只有内存 `exec_queue` 和后台 `asyncio.create_task()`，无持久化队列、重试、优先级、任务状态表、用户级隔离。
2. **任务调度器不具备资源调度**：没有 GPU/CPU/容器/节点级排队与资源分配；仅有可选 Docker 和可视化 subprocess。
3. **算法仓库概念分散**：算法散落在 `TASK_REGISTRY`、`MODEL_REGISTRY`、`TOOLS`、`skills`；没有统一元数据、版本、I/O schema、资源需求和评测状态。
4. **算法评测器不统一**：训练评测、工具打分、workflow 评价分散，缺少“新算法接入”标准流程。
5. **应用层不完整**：后端 API 与前端 JSON parser 存在，但仓库没有完整前端应用代码。
6. **合成规划/临床优化主要在 skills 或组合层**：core tools 里没有完整专用执行器。
7. **接口校验有不一致点**：例如 server 的 `import_pocket` 配置声明 required input 是 `pocket, indices`，但 handler 实际读取 `protein, indices`；建议修正为 `protein, indices`。

### 5.3 如果要演进到参考图架构

建议新增或强化：

- `AlgorithmSpec`：统一算法元数据（name、domain、version、inputs、outputs、resource、container、metrics）。
- `AlgorithmRegistry`：把 `TASK_REGISTRY`、`TOOLS`、`skills` 统一到可检索仓库。
- `TaskQueue`：持久化任务表、状态机、重试、优先级、用户配额。
- `Scheduler`：按资源需求把任务发到本机、Docker、K8s、远程 GPU worker。
- `Evaluator`：统一 benchmark 数据集、指标、报告、可视化和回归测试。
- `DataConnector`：将内部数据对象、外部数据库、文件存储、OSS URL 做成一致 schema。
- `WorkflowRuntime`：将当前 YAML DAG 扩展为可恢复、可追踪、可缓存、可审计的 pipeline runtime。

## 6. 快速索引：核心文件

| 文件 | 作用 |
|---|---|
| `open_biomed/core/agent.py` | LLM 任务规划、工具调用、代码执行、报告导出 |
| `open_biomed/core/workflow.py` | 前端图解析、Workflow DAG 构建与执行、边传递 |
| `open_biomed/core/pipeline.py` | 训练/验证 Pipeline、推理 Pipeline、Ensemble Pipeline |
| `open_biomed/tools/tool_registry.py` | 所有可调用工具的懒加载注册表 |
| `open_biomed/tools/base_tool.py` | Tool 抽象接口与 serial execution 装饰器 |
| `open_biomed/tools/web_request_tools.py` | 外部数据库/搜索请求工具 |
| `open_biomed/tools/third_party_tools.py` | 第三方 CLI 工具，例如 P2Rank |
| `open_biomed/tools/tool_misc.py` | pocket 导入、导出、突变应用、摘要、PDB 分子提取等工具 |
| `open_biomed/scripts/inference.py` | 重模型推理工具构造函数 |
| `open_biomed/scripts/run_server.py` | FastAPI 工具服务 |
| `open_biomed/scripts/run_server_workflow.py` | FastAPI workflow/report 服务 |
| `open_biomed/tasks/__init__.py` | 支持的 task registry |
| `open_biomed/data/*` | 运行时数据对象 |
| `configs/workflow/*`、`memory/workflows/*` | 预定义 workflow |
| `skills/*/SKILL.md` | 高层 biomedical skills/算法工作流说明 |

## 7. PlannerExecutor 完整调用链：Prompt → Tool Selection → Workflow Execution

`PlannerExecutor` 是 OpenBioMed 中最接近“任务规划器 / 智能路由”的实现，完整链路如下：

```text
agent_cfg
  -> PlannerExecutor.__init__
      -> get_llm(agent_cfg.llm, stop_sequences=["</execute>", "</report>"])
      -> CONTEXT_MANAGERS[name](context_manager_cfg)
      -> _init_tools(): self.tools = TOOLS
      -> _init_memory(): optionally self.workflows = WORKFLOWS
      -> _init_system_prompt(): 拼接规划格式、执行协议、工具说明、workflow memory
      -> _setup_execution_environment(): optionally Docker
      -> _init_agent_workflow(): LangGraph generate/execute 状态机
  -> run(user_prompt, thread_id=None)
      -> context_manager.reinitialize(SystemMessage(prompt))
      -> context_manager.add_message(HumanMessage(user_prompt))
      -> app.stream(...)
          -> generate(): LLM 读取上下文并产出 <execute> 或 <report>
          -> execute(): 执行 Python/Bash 代码，把 observation 回写上下文
          -> generate(): 基于 observation 继续选择工具/执行下一步
      -> return thread_id, messages, captured_results
```

### 7.1 Prompt 组装

Prompt 中包含四类关键信息：

1. **角色与规划协议**：要求先制定 checklist 或 step-by-step table，然后逐步执行，并在每步后更新计划。
2. **代码执行协议**：LLM 必须输出 `<execute>...</execute>` 或 `<report>...</report>`；`<execute>` 中可写 Python，或以 `#!BASH` / `#!CLI` 开头执行 shell。
3. **Tool selection 信息**：如果未启用 `tool_retriever`，系统会遍历 `self.tools.available_tools()`，把每个工具的 `print_usage()` 写入 prompt。因此当前的 tool selection 是 **LLM 基于工具说明自行选择**，不是硬编码路由器或向量检索器。
4. **Workflow memory 信息**：如果配置 `memory.workflow=True`，系统把 `WORKFLOWS` 中每个 workflow 的 `metadata`（description、expected inputs、expected outputs）写入 prompt，使 LLM 可选择直接调用预定义 workflow。

### 7.2 Tool Selection 到 Tool Call

当前支持两种调用方式：

| `tool_call` 模式 | Prompt 中给 LLM 的调用方式 | 输入转换 | 输出约定 |
|---|---|---|---|
| `custom` | Python 中 `from open_biomed.tools.tool_registry import TOOLS`，再 `TOOLS[tool_name].run(...)` | 对 Molecule/Protein/Pocket/Text 用 `create_tool_input(data_type, value)` 构造对象 | Tool 返回 `(outputs, observations)` 两个等长列表 |
| `http_request` | Python 中 `requests.post(f"{OPENBIOMED_SERVICE_URL}/tool/...")` | 分子/蛋白/口袋/文本以路径、SMILES、FASTA 或字符串传入 | JSON 中包含 `outputs` 与 `observations` |

### 7.3 Workflow Execution 调用

当 Agent 选择 workflow 时，典型代码链如下：

```python
from open_biomed.core.workflow import WORKFLOWS
workflow = WORKFLOWS["pdb_query"]
result, messages = workflow.run(
    inputs=[(0, "accession", "4xli"), (1, "accession", "4xli")],
    num_repeats=1,
)
```

其运行结果约定：

- `result[i][j][k]`：第 `i` 次重复、第 `j` 个输出节点、第 `k` 个输出对象。
- `messages[i]`：第 `i` 次重复的 workflow 执行过程说明。

因此，完整箭头可以概括为：

```text
用户需求 Prompt
  -> LLM 生成计划
  -> LLM 根据 tool/workflow usage 选择工具或 workflow
  -> <execute> Python/Bash
      -> TOOLS[name].run(...) 或 WORKFLOWS[name].run(...)
      -> Tool/Workflow outputs + observations
  -> observation 回写给 LLM
  -> 下一轮 tool selection / workflow execution
  -> <report> 最终报告
```

## 8. `tool_registry.py`：Tool 注册机制与动态加载机制

`tool_registry.py` 的核心是 `LazyDictForTool(dict)` 与全局实例 `TOOLS`。

### 8.1 注册机制

| 机制 | 说明 |
|---|---|
| 顶部导入 | `tool_registry.py` 先导入 `tool_misc`、`web_request_tools`、`visualization_tools`、`third_party_tools`、`open_biomed.scripts.inference` 等模块，使具体 Tool 类和 `test_*` pipeline factory 可见。 |
| `available_tools()` | 返回一个固定字符串列表，作为 Agent prompt、API 服务与用户可见工具集合。 |
| `__missing__(self, key)` | 当访问 `TOOLS[key]` 且 key 不在字典中时，根据 `key` 分支创建具体 Tool 实例。 |
| 全局 `TOOLS` | `TOOLS = LazyDictForTool()`；调用方不直接实例化工具，而是通过 `TOOLS[tool_name]` 获取。 |

### 8.2 动态加载机制

这里的“动态加载”主要是 **懒加载（lazy initialization）**，不是自动扫描目录注册插件：

1. 首次访问 `TOOLS["molecule_property_prediction"]`。
2. `dict.__getitem__` 未找到 key，触发 `LazyDictForTool.__missing__`。
3. `__missing__` 根据 key 调用具体构造逻辑，例如：
   - 重模型推理工具：`test_molecule_property_prediction(unit_test=False)` 返回 `InferencePipeline` 或 `EnsemblePipeline`。
   - 外部请求工具：`PubChemRequester()`、`UniProtRequester()`、`PDBRequester()`、`STRINGRequester()`。
   - 可视化工具：`PyMolVisualizerWrapper(task="visualize_molecule")` 等。
   - 轻量工具：`MutationToSequence()`、`ImportPocket()`、`ExportMolecule()`、`MoleculeQEDTool()` 等。
4. 实例被缓存到 `self[key]`，后续访问复用同一对象。
5. 未支持 key 抛出 `NotImplementedError`。

### 8.3 Tool 分类

| 类别 | 典型 key | 实例来源 |
|---|---|---|
| 重模型推理 | `text_based_molecule_editing`, `structure_based_drug_design`, `protein_folding`, `mutation_engineering` | `open_biomed/scripts/inference.py` 中 `test_*` 函数构造 `InferencePipeline` |
| 外部数据库/搜索 | `molecule_name_request`, `protein_uniprot_request`, `protein_pdb_request`, `ppi_string_request`, `chembl_query`, `web_search` | `web_request_tools.py` 中 requester 类 |
| 第三方 CLI | `protein_binding_site_prediction` | `ProteinBindingSitePrediction` 调用 P2Rank CLI |
| 可视化 | `visualize_molecule`, `visualize_protein`, `visualize_complex`, `visualize_protein_pocket` | `PyMolVisualizerWrapper` |
| 轻量工具/规则工具 | `apply_mutation_to_sequence`, `import_pocket`, `export_molecule`, `molecule_qed`, `molecule_sa`, `molecule_logp`, `molecule_lipinski`, `molecule_similarity` | `tool_misc.py`、`open_biomed.data.molecule` 中 Tool 类 |

## 9. `workflow.py`：DAG 执行流程与 `name_mapping` 实现

### 9.1 Workflow 配置 Schema

内部 workflow YAML 的核心结构：

```yaml
metadata:
  description: ...
  inputs:
    - [node_id, input_name, input_description]
  outputs:
    - [node_id, output_description]
tools:
  - name: protein_pdb_request
    inputs:
      mode: metadata
  - name: code_execution
    code: |
      ...
edges:
  - start: 0
    end: 1
    name_mapping:
      output: pdb_file
```

其中：

- `tools[i].name` 对应 `TOOLS[name]`；`code_execution` 是特殊节点，执行 YAML 中的 Python code。
- `tools[i].inputs` 是节点初始输入，会通过 `create_tool_input()` 转成对象。
- `edges` 定义 DAG 中上游节点 `start` 到下游节点 `end` 的数据流。
- `name_mapping` 是边上的可选字段，用于把上游输出字段名重命名为下游需要的输入字段名。

### 9.2 DAG 构建

`Workflow(config)` 初始化流程：

1. 读取 `config.metadata`，拼出 workflow 描述与 expected inputs。
2. 遍历 `config.tools`：
   - `name == "code_execution"` 时创建 `WorkflowNode(name="code_execution", executable=exec, inputs={"code": ...})`。
   - 否则创建 `WorkflowNode(name=tool["name"], executable=TOOLS[tool["name"]], inputs=tool.get("inputs", {}))`。
3. 遍历 `config.edges`，保存 `(start, end, name_mapping)`。
4. 计算出度为 0 的节点作为默认 `output_nodes`；若 `metadata.outputs` 存在，则使用显式输出节点。

### 9.3 DAG 执行流程

`Workflow.run(inputs, num_repeats, deamon)` 的执行流程：

```text
run(inputs)
  -> _initial_state(inputs)
      -> 计算每个节点 in_deg
      -> 重置 node.inputs = node.orig_inputs
      -> 将外部 inputs[(node_idx, input_name, value)] 注入节点
      -> exec_queue = 所有 in_deg == 0 的节点
  -> while determine_next_step(state):
      -> step(state)
          -> exec_queue.sort(); pop 最小 node_idx
          -> 对 node.num_repeats 循环执行
              -> async tool: asyncio.run(tool.run(...))
              -> code_execution: exec(code, {step_inputs, step_outputs})
              -> normal tool: tool.run(**node.inputs)
          -> node.outputs 累积 outputs 与 messages
          -> 遍历所有从 node_idx 出发的边
              -> 下游 in_deg -= 1
              -> wrap_outputs(node.outputs[0]) 得到 {molecule/protein/pocket/text/output: value}
              -> 应用 name_mapping 改字段名
              -> 写入 downstream_node.inputs[key] = value
              -> 若下游 in_deg == 0，则加入 exec_queue
          -> state.messages 追加 node.pretty_print(...)
  -> 收集 output_nodes 的 outputs
  -> return results, messages
```

### 9.4 `name_mapping` 实现细节

`wrap_outputs()` 会按对象类型生成默认字段名：

| 上游输出对象 | 默认字段名 |
|---|---|
| `Molecule` | `molecule` |
| `Protein` | `protein` |
| `Pocket` | `pocket` |
| `Text` | `text` |
| 其他类型 | `output` |

如果边上配置了 `name_mapping`，执行时会检查：

```python
if e[2] is not None and key in e[2]:
    key = e[2][key]
```

典型用法：

| 场景 | 上游默认字段 | 下游需要字段 | 配置 |
|---|---|---|---|
| PDB 文件提取分子 | `output` | `pdb_file` | `name_mapping: {output: pdb_file}` |
| 突变工程输出接突变应用 | `output` | `mutation` | `name_mapping: {output: mutation}` |

因此，`name_mapping` 是 OpenBioMed workflow 中连接“非标准输出字段”和“下游工具参数名”的关键箭头机制。

## 10. `pipeline.py`：TrainValPipeline 与 InferencePipeline

### 10.1 `TrainValPipeline`

`TrainValPipeline` 面向训练、验证和测试，调用链如下：

```text
CLI args
  -> Config(config_file="./configs/basic_config.yaml", **args)
  -> merge_config(additional_config_file...)
  -> TASK_REGISTRY[cfg.task]
  -> task.get_monitor_cfg()
  -> setup_infra()
      -> logs/{task}/{model}-{dataset}/{exp_name}
      -> checkpoint_dir / val_outputs / test_outputs
      -> seed / distributed / wandb logger
  -> setup_model()
      -> task.get_model_wrapper(cfg.model, cfg.train)
      -> ModelCheckpoint / pl.Trainer callbacks
  -> setup_data()
      -> task.get_datamodule(cfg.dataset, *model.get_featurizer())
  -> run()
      -> trainer.fit(...)
      -> trainer.test(...)
```

关键接口：

| 阶段 | 输入 | 输出 |
|---|---|---|
| 配置解析 | CLI args + basic config + additional config | `self.cfg` |
| Task 选择 | `self.cfg.task` | `TASK_REGISTRY[task]` 对应的 Task 类 |
| 数据准备 | dataset config + featurizer + collator | Lightning datamodule，含 train/val/test dataloader |
| 模型准备 | model config + train config | Lightning `ModelWrapper` |
| 训练/测试 | dataloaders + checkpoint | checkpoints、logs、test metrics |

### 10.2 `InferencePipeline`

`InferencePipeline` 同时继承 `Pipeline` 和 `Tool`，是重模型工具接入 `TOOLS` 的核心封装。

调用链：

```text
InferencePipeline(task, model, model_ckpt, additional_config, device, output_prompt)
  -> 读取 configs/model/{model}.yaml
  -> 可选 merge additional_config
  -> cfg.task = task; cfg.model_ckpt = model_ckpt; cfg.device = device
  -> TASK_REGISTRY[task]
  -> run(**kwargs)
      -> lazy setup_infra()
      -> lazy setup_model()
          -> task.get_model_wrapper(cfg.model, None)
          -> model.get_featurizer() -> featurizer, collator
          -> load checkpoint if exists
          -> eval + to(device)
      -> 将 kwargs 每个字段转为 list
      -> 对每个样本调用 featurizer(**inputs)
      -> batch_size=max/auto/int
      -> collator(inputs)
      -> model.transfer_batch_to_device(...)
      -> model.predict(...)
      -> 对 None 输出按 retry_limit 重试
      -> Molecule/Protein/Pocket 自动保存为 ./tmp/*.pkl
      -> return outputs_or_prompt_texts, files
```

关键接口：

| 输入 | 说明 |
|---|---|
| `task` | 必须在 `TASK_REGISTRY` 中，例如 `protein_folding`、`structure_based_drug_design` |
| `model` | 对应 `configs/model/{model}.yaml` |
| `model_ckpt` | checkpoint 路径；存在时加载 |
| `additional_config` | 可叠加数据集/任务配置 |
| `device` | `cpu` 或 `cuda:*` |
| `kwargs` | 任务输入对象，例如 `molecule`, `protein`, `pocket`, `text`, `mutation`, `go_terms` |

输出：

| 输出形式 | 说明 |
|---|---|
| `outputs` | 模型预测出的 Python 对象或数值/文本 |
| `files` | 对 `Molecule`/`Protein`/`Pocket` 自动保存 `.pkl`；其他输出转字符串 |
| `output_prompt` | 如果配置了 prompt，则返回格式化后的自然语言文本作为 outputs |

### 10.3 `EnsemblePipeline`

`EnsemblePipeline` 持有 `Dict[str, InferencePipeline]`，运行时从输入 `kwargs` 中弹出 `task`，并调用 `self.pipelines[task].run(**kwargs)`。当前 `molecule_property_prediction` 用它把 BBBP、SIDER、Caco2、half-life、LD50 等多个性质预测 pipeline 聚合到一个工具入口。

## 11. `TASK_REGISTRY`：所有任务类型与输入输出 Schema

`TASK_REGISTRY` 是 OpenBioMed 的任务到 Task 类的中心映射。`TrainValPipeline` 和 `InferencePipeline` 都通过它完成 task name 到任务实现的解析。

| Task key | Task 类 | 输入 Schema | 输出 Schema | 备注 |
|---|---|---|---|---|
| `text_based_molecule_editing` | `TextMoleculeEditing` | `molecule: Molecule`, `text: Text`（期望改造性质） | `Molecule`（结构相似但性质按文本改进的新分子） | 分子编辑 |
| `molecule_captioning` | `MoleculeCaptioning` | `molecule: Molecule` | `Text` / 分子详细文本描述 | 分子到文本 |
| `text_guided_molecule_generation` | `TextGuidedMoleculeGeneration` | `text: Text`（目标分子描述） | `Molecule` | 文本到分子 |
| `molecule_question_answering` | `MoleculeQA` | `molecule: Molecule`, `text: Text`（问题） | `Text` / answer | 分子问答 |
| `protein_question_answering` | `ProteinQA` | `protein: Protein`, `text: Text`（问题） | `Text` / answer | 蛋白问答 |
| `text_based_protein_generation` | `TextBasedProteinGeneration` | `text: Text`（目标蛋白性质说明） | `Protein` sequence or structure | 文本到蛋白 |
| `molecule_property_prediction` | `MoleculePropertyPrediction` | `molecule: Molecule`；在 ensemble/server 中还传 `task`/`dataset` 选择性质 | `float` / property score | 分类性质预测 |
| `molecule_property_prediction_regression` | `MoleculePropertyPredictionRegression` | `molecule: Molecule` | `float` | 回归性质预测 |
| `pocket_molecule_docking` | `PocketMoleculeDocking` | `molecule: Molecule`, `pocket: Pocket` | `Molecule`（含 `.conformer` 结合构象） | 配体-口袋对接 |
| `structure_based_drug_design` | `StructureBasedDrugDesign` | `pocket: Pocket` | `Molecule`（可能结合口袋的新分子） | 基于结构药物设计 |
| `structure_text_based_molecule_optimization` | `StructureTextBasedMoleculeOptimization` | `molecule: Molecule`, `pocket: Pocket`, `text: Text` | `Molecule`（兼顾口袋结合与文本目标） | 结构+文本分子优化 |
| `mutation_explanation` | `MutationExplanation` | `protein: Protein`, `mutation: str`（如 `A51F`） | `Text` / 突变影响说明 | 突变解释 |
| `mutation_engineering` | `MutationEngineering` | `protein: Protein`, `text: Text` / prompt（期望突变性质） | `str` / mutation string 或 mutation list | 突变设计 |
| `protein_folding` | `ProteinFolding` | `protein: Protein`（通常需要 sequence） | `Protein`（带 3D 结构） | 蛋白折叠 |
| `cell_annotation` | `CellAnnotation` | `cell: Cell`, `class_texts: List[Text]` | `int`（预测细胞类型索引） | 单细胞注释 |
| `go_guided_protein_generation` | `GoGuidedProteinGeneration` | `go_terms: List[str]` 或 List[List[str]] | `Protein` sequence | GO 引导蛋白生成 |

## 12. 补充后的架构箭头总览

结合以上补充，OpenBioMed 当前最完整的端到端链路可以写成：

```text
[应用层 / 用户]
  -> FastAPI TaskRequest 或 PlannerExecutor user_prompt
  -> PlannerExecutor Prompt 组装
      -> Tool usage + Workflow metadata 注入
      -> LLM tool/workflow selection
  -> tool_registry.TOOLS 懒加载 Tool
      -> 外部 Requester / 轻量 Tool / InferencePipeline / EnsemblePipeline / 第三方 CLI
  -> InferencePipeline 或 Tool.run
      -> TASK_REGISTRY[task]
      -> Task.get_model_wrapper / get_featurizer / collator / model.predict
  -> Tool outputs, observations
  -> Workflow DAG
      -> wrap_outputs 按对象类型命名
      -> name_mapping 重命名边输出字段
      -> 下游节点输入注入
  -> output_nodes / Agent observation / API response / report
```

与参考架构相比，OpenBioMed 已经具备“规划器 + 工具注册 + DAG workflow + 数据对象 + 若干执行器”的原型闭环；仍缺生产级任务队列、统一资源调度、统一算法仓库元数据与统一算法评测器。
