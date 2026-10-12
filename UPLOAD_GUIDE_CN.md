# GitHub 上传说明

这是从恢复文件中选出的待发布整理包。22 个 Python 文件的原始内容未改动；已补英文 README、依赖清单、数据格式说明、固定空间划分和部分最终统计结果。它还不是已完成真实数据复现的投稿仓库。

## 现在上传哪些

建议新建仓库 `GFSC-UNet-HM`，将本文件夹内的文件和子目录作为仓库内容，保持 Python 文件在同一层。保留 `docs/source_manifest.csv`，其中列明每个恢复来源文件的用途和 SHA-256。

主模型最小代码集合是：`gfsc_unet_d3_hybrid_mask_main.py`、`rh4_final_strict_light.py`、`fusion_final_strict.py`、`hybrid_masking.py`。其余已选脚本用于基线、消融、密度压力、空间分块、多种子、统计与作图。

D3/RM 实验代码仍是当前消融及匹配种子对照所需依赖，不能因为名称中含有 D3 就全部删除。旧 `rh.py/rh1.py/rh2.py/rh3.py`、`cjunet*.py`、旧模型输出、日志、进程号、缓存和开题/修改记录没有打入此包。

## 必须从服务器补齐

- `f3_direct_ablation_final_strict_v1/D3/protocol_no2.json`
- `f3_direct_ablation_final_strict_v1/D3/protocol_o3.json`
- `f3_direct_ablation_final_strict_v1/D3/protocol_pm25.json`
- 实际生成最新 `duoyuan/data` 的预处理代码、输入字段/单位/顺序说明。
- 最新空间异常及站点对浓度差分析的生成脚本。当前恢复的 `spatial_skill_analysis.py` 是较早诊断，不是最终站点对差异分析的替代品。
- 论文最终逐站预测、匹配 HM/RM 三种子结果和原始运行环境版本。

大体积原始数据和权重可提供数据托管或发布下载链接；是否能公开原始数据须按现有数据使用权限处理。能直接共享的数据与有限制数据应在数据可用性说明中分别交代。

## 投稿前

补许可证、英文注释与真实数据/兼容示例；处理本机默认路径并进行干净环境试跑。README 内已有主程序的 preflight 命令。`run_rm_multiseed_spatial.py` 对四个来源文件执行 SHA-256 校验，当前打包文件已核对一致；修改注释或默认路径后需要同步记录版本并更新校验，不能仅删除检查。

压缩包供你下载和解压。GitHub 应上传解压后的源文件和目录；C&G 当前要求不能用单个 ZIP 替代可浏览的公开代码仓库。
