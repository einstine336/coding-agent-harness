# 代码修复评测

自由补丁评测让 Agent 阅读合成 Python 项目、编写补丁并接受测试检查。完整修复要求缺陷复现测试和原有行为回归测试均通过。隐藏检查用于最终评分。

三组消融共享首轮补丁，对比不重试、只告知失败状态后重试和提供完整失败诊断后重试。常规任务从 Agent 写出的补丁开始，受控任务从预先构造的不完整补丁开始。模型生成的补丁在 WASI Python 运行时内执行。

在项目根目录运行。下面的命令把评测依赖安装到独立目录。在线评测使用本地 `.env` 配置的 DeepSeek Anthropic 兼容接口。完成后会在本地输出逐题结果。

```powershell
$py = '.\.venv\Scripts\python.exe'
uv pip install --only-binary :all: --target .\eval\.wasi-deps --python $py 'wasi-python==0.1.4' 'wasmtime==50.0.1'
$env:PYTHONPATH = (Resolve-Path .\eval\.wasi-deps).Path
& $py -B .\eval\run_repair_ablation.py --check-only
& $py -B .\eval\run_repair_ablation.py --env-file .env --output .\eval\ablation_rerun.json
```

中断后可在相同参数下加 `--resume` 继续。运行时与测试验收的本地检查位于本目录的测试文件中。
