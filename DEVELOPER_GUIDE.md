# Chinatravel-Main 开发者文档

## 1. 项目概览
Chinatravel-Main 是为旅行规划竞赛（Travel Planning Challenge, TPC）准备的本地推理与评测框架。项目提供了：

- 旅行沙盒环境的完整数据（景点、交通、餐饮等）。
- 用于加载 LLM/算法智能体的运行脚本与抽象基类。
- 评测旅行计划可行性、约束满足度以及偏好指标的评估流水线。
- 参考的本地模型下载脚本与占位实现，方便参赛者集成自定义模型。

本文档面向开发者，详细解释每个文件夹和关键文件的作用，帮助快速理解项目结构与二次开发流程。

## 2. 顶层目录结构

| 路径 | 说明 |
| --- | --- |
| `README.md` | 项目简述、竞赛阶段说明以及运行/评估命令概览。|
| `DEVELOPER_GUIDE.md` | 本文档，介绍项目结构与各模块用途。|
| `requirements.txt` | Python 依赖列表，包含地理计算、LLM、评测等所需库。|
| `download_llm.sh` | 使用 ModelScope 下载 Qwen3-8B 本地模型的脚本。|
| `run_tpc.py` | 运行旅行规划智能体的入口脚本，负责加载查询、初始化环境与模型并批量生成方案。|
| `eval_tpc.py` | 评测脚本，调用评估模块对生成方案进行结构/常识/硬约束/偏好等指标计算。|
| `chinatravel/` | 主要 Python 包，包含 agent、数据、环境、评测、符号验证及本地模型等子模块。|

## 3. `chinatravel/` 包结构

```
chinatravel/
├── agent/
├── data/
├── environment/
├── evaluation/
├── local_llm/
└── symbol_verification/
```

### 3.1 `chinatravel/agent/`
负责定义智能体与语言模型的抽象接口、工具函数以及 TPC 专用 agent。

- `base.py`
  - `AgentReturnInfo`、`AbstractAgent`、`BaseAgent` 等基类，约定智能体与环境交互的接口和返回格式。【F:chinatravel/agent/base.py†L1-L153】
  - 包含日志目录创建、计时等通用逻辑，子类只需重写 `run`、`reset` 等方法。
- `llms.py`
  - 定义 `AbstractLLM` 抽象类以及 DeepSeek、GLM4、GPT-4o、Qwen、Mistral、Llama 等多种 LLM 适配器，负责统一的对话调用接口与 token 统计。【F:chinatravel/agent/llms.py†L1-L200】
  - 实现 JSON 修复、停用词控制等推理辅助功能。
- `load_model.py`
  - `init_agent`：根据命令行参数实例化不同策略的智能体（规则、LLM 驱动、ReAct、TPC 专用等）。【F:chinatravel/agent/load_model.py†L1-L61】
  - `init_llm`：根据名称创建对应的 LLM 适配器，支持加载本地 `TPCLLM` 模板。【F:chinatravel/agent/load_model.py†L63-L91】
- `utils.py`
  - JSON 与 NumPy 数据的编码/解码工具、简单日志类 `Logger` 等常用工具函数。【F:chinatravel/agent/utils.py†L1-L62】
- `tpc_agent/`
  - `tpc_agent.py`：TPC 自定义智能体占位类 `TPCAgent`，继承 `BaseAgent`，在 `run` 方法中重置计时并返回规划结果结构，供参赛者填充具体策略。【F:chinatravel/agent/tpc_agent/tpc_agent.py†L1-L34】
  - `tpc_llm.py`：自定义本地 LLM 模板，继承 `AbstractLLM`，默认返回占位响应，方便开发者接入真实模型。【F:chinatravel/agent/tpc_agent/tpc_llm.py†L1-L21】

### 3.2 `chinatravel/data/`
提供任务查询数据集的读取与保存逻辑。

- `load_datasets.py`
  - 使用 HuggingFace `datasets` 或本地 JSON 读取比赛查询，依据 `--splits` 选择数据范围。【F:chinatravel/data/load_datasets.py†L1-L130】
  - 支持 `--oracle_translation` 参数决定是否保留硬逻辑约束（例如 DSL）。
  - 提供 `save_json_file`、`load_json_file` 等通用 JSON 读写函数。
- `tpc_aic_phase/`
  - 目录下大量 `*.json` 文件为官方提供的查询样本，命名为查询 UID。参赛者运行脚本时会根据分割列表加载这些文件。

### 3.3 `chinatravel/environment/`
实现沙盒环境的数据、API 适配器与世界环境封装。

- `world_env.py`
  - 定义 `WorldEnv` 类，通过字符串形式调用内部 API（如 `attractions_select(...)`）。
  - `EnvOutput` 封装返回结构，支持 DataFrame 分页和文本化展示。【F:chinatravel/environment/world_env.py†L1-L211】
- `tool/`
  - 汇总各种环境查询接口，`__init__.py` 暴露 `Attractions`、`Accommodations`、`Restaurants`、`IntercityTransport`、`Transportation`、`Poi` 等类。【F:chinatravel/environment/tool/__init__.py†L1-L15】
  - `attractions/apis.py`：加载景点 CSV 数据，提供按条件筛选、开放时间校验、附近景点查询等方法。【F:chinatravel/environment/tool/attractions/apis.py†L1-L97】
  - `accommodations/apis.py`：酒店查询接口，支持字段过滤、附近酒店查询、字段类型查看等。【F:chinatravel/environment/tool/accommodations/apis.py†L1-L85】
  - `restaurants/apis.py`：餐厅数据的读取与查询，结构与景点/酒店类似（CSV 数据 + 附近检索 + 营业状态等）。
  - `intercity_transport/apis.py`：跨城交通信息（航班/火车），提供根据条件筛选、票价/时刻查询等 API。
  - `transportation/apis.py`：城市内交通（地铁、打车、步行）模型，内置地铁图构建、Dijkstra 最短路径、费用估算等函数。【F:chinatravel/environment/tool/transportation/apis.py†L1-L135】
  - `poi/apis.py`：兴趣点经纬度索引，可根据名称检索城市坐标，用于附近检索与交通估算。【F:chinatravel/environment/tool/poi/apis.py†L1-L60】
- `database/`
  - `attractions/`、`accommodations/`、`restaurants/`、`intercity_transport/`、`transportation/`、`poi/` 等数据目录，保存各城市 CSV/JSON 数据。
  - 例如 `database/transportation/subways.json` 存储城市地铁线路与车站信息；`restaurants/<city>/restaurants_<city>.csv` 提供餐饮列表。

### 3.4 `chinatravel/evaluation/`
封装旅行计划的结构校验、常识/逻辑约束检测与偏好打分逻辑。

- `output_schema.json`：旅行计划 JSON 的 Schema，校验字段合法性。
- `utils.py`
  - 提供 JSON Schema 验证、文件读写，以及继承自环境 API 的 `AttractionsOODTag`（带 Out-of-domain 标注的景点数据，用于评估）。【F:chinatravel/evaluation/utils.py†L1-L77】
- `commonsense_constraint.py`：调用 `symbol_verification` 中的检查函数，评估交通、景点、酒店、餐饮、时间、空间等常识约束，通过率输出宏/微平均得分。【F:chinatravel/evaluation/commonsense_constraint.py†L1-L83】
- `hard_constraint.py` / `hard_constraint_v2`：组合逻辑约束，确保旅行计划满足 DSL/逻辑表达的必达条件。
- `schema_constraint.py`：基于 JSON Schema 的结构验证，输出通过率统计。
- `preference.py`：计算旅行偏好指标（景点数量、交通时长、餐饮推荐等），与 `DEFAULT_PR` 脚本一致，用于最终综合评分。
- `rank.py`：汇总评估结果，对多种方法进行排序比较。
- `default_splits/`：包含 `tpc_phase_1_train.txt`、`tpc_phase_1_test.txt` 等划分文件，用于从本地数据集中选择查询样本。

### 3.5 `chinatravel/local_llm/`
存放本地语言模型权重或分词器。

- `Qwen3-8B/`：下载占位目录（脚本会填充模型文件）。
- `deepseek_v3_tokenizer/`：DeepSeek 分词器配置（`tokenizer.json`、`tokenizer_config.json`、`deepseek_tokenizer.py` 等）。

### 3.6 `chinatravel/symbol_verification/`
包含常识、逻辑、偏好验证所需的符号化函数。

- `commonsense_constraint.py`、`hard_constraint.py`、`preference.py`：实现具体的判定逻辑，被评估模块调用。
- `concept_func.py`：提供 DSL/逻辑约束中用到的概念函数映射，在评估偏好或条件逻辑时动态注入。
- `readme.md`：罗列各类环境/个人约束的含义与判定标准，方便理解验证维度。【F:chinatravel/symbol_verification/readme.md†L1-L33】

## 4. 运行脚本说明

### 4.1 `run_tpc.py`
- 解析命令行参数：选择查询切分 (`--splits`)、指定智能体与 LLM（`--agent`, `--llm`）、是否使用 Oracle 翻译等。【F:run_tpc.py†L1-L78】
- 调用 `load_query` 加载查询数据，初始化 `WorldEnv` 与 LLM 实例，然后通过 `init_agent` 创建智能体。
  - 迭代查询样本，调用智能体的 `run` 方法生成规划结果，并将结果保存至 `results/<Agent>_<LLM>/` 下的 JSON 文件。【F:run_tpc.py†L80-L125】

### 4.2 `eval_tpc.py`
- 支持根据 `--splits`、`--method` 读取指定目录中的规划结果。
- 先执行 JSON Schema 校验，再依次评估常识约束、硬约束、偏好指标，并将得分写入 `eval_res/` 下对应方法的目录。【F:eval_tpc.py†L1-L214】
- 提供 `DEFAULT_PR` 脚本示例，展示如何自定义偏好函数并计算平均得分。

## 5. 辅助文件

- `requirements.txt`：列出运行所需的 Python 包及版本（例如 `geopy`、`pandas`、`transformers`、`vllm`、`torch` 等），安装后即可运行环境 API 与模型适配代码。【F:requirements.txt†L1-L25】
- `download_llm.sh`：依赖 ModelScope CLI，一键下载 Qwen3-8B 模型至 `chinatravel/local_llm/Qwen3-8B/` 目录。【F:download_llm.sh†L1-L1】

## 6. 开发建议

1. **定制智能体**：在 `chinatravel/agent/tpc_agent/` 中扩展 `TPCAgent`，利用 `WorldEnv` 提供的 API 完成规划生成。
2. **集成模型**：实现 `TPCLLM._get_response` 或者在 `init_llm` 中添加新模型，确保符合 `AbstractLLM` 接口。
3. **数据理解**：熟悉 `environment/database/` 下的数据格式，尤其是字段命名、单位等，确保规划输出与环境一致。
4. **本地评测**：通过 `run_tpc.py` 生成结果，再用 `eval_tpc.py` 校验，及时观察 `eval_res/` 中的得分与错误日志。
5. **符号验证扩展**：如需添加自定义逻辑，可在 `symbol_verification/` 中扩展概念函数或约束判定，并在评估模块注册。

通过本指南，开发者可以快速定位需要修改的模块，理解项目内各目录/文件的功能，从而高效地完成旅行规划算法的设计与评测。
