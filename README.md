# CodeLoop｜智能编码与自动验收 Harness

CodeLoop 是一个运行在终端里的 Python Coding Agent。给它一个代码变更任务和一个 pytest 用例，它会检查代码、生成补丁，并根据测试结果判断是否需要再修一次。

- **Agent Loop 与多 Agent**。统一处理工具调用、权限检查与结果回写。支持 OpenAI 兼容和 Anthropic 兼容接口、Plan Mode、MCP，以及在独立上下文中执行任务的子 Agent。
- **测试验收**。根据 pytest 用例 ID 验证代码变更。用例失败时将诊断交回 Agent，可限制追加修复次数。跳过、未收集和超时会标记为未验证。
- **会话与 Skills**。支持恢复历史会话、折叠长对话、召回项目记忆，以及从用户反馈中抽取和评估可复用的技能。

## 快速开始

需要 Python 3.11 或更新版本。以下是 PowerShell 示例。在配置模型密钥前，可以用本地模拟模型验证 Agent 是否能完成工具调用并读到文件。

```powershell
git clone https://github.com/einstine336/coding-agent-harness.git
Set-Location coding-agent-harness
python -m venv .venv
$codeLoopRoot = (Get-Location).Path
$codeLoopPython = Join-Path $codeLoopRoot '.venv\Scripts\python.exe'
& $codeLoopPython -m pip install -r requirements.txt
& $codeLoopPython tools/verify_offline.py
```

## 在自己的项目里验证代码变更

从 CodeLoop 仓库根目录设置 Python 模块路径，再切换到目标项目根目录。参考 [配置模板](.env.example)，在目标项目的 `.env` 中填入模型配置与自己的 API 密钥。把下面的用例 ID 换成目标项目中实际存在的用例。

```powershell
$env:PYTHONPATH = $codeLoopRoot
Set-Location '<目标项目目录>'
& $codeLoopPython -m agents.main --verify 'tests/test_pricing.py::test_discount' '修复折扣边界问题'
```

验收默认使用启动 Agent 的 Python。目标项目有独立的虚拟环境时，使用 `--test-python <解释器路径>` 指定能导入 pytest 和项目依赖的解释器。在交互模式下，`/verify <用例 ID>` 可单独运行测试。

## 评测方法

用缺陷复现测试检查补丁是否生效，用回归测试检查已有行为是否保持。对比不重试、收到失败状态后重试和收到完整诊断后重试三种流程。数据和命令见 [评测说明](eval/README.md)。项目也提供 [Docker 构建文件](Dockerfile)。

## 来源与许可

运行时基于 [Windy3f3f3f3f/claude-code-from-scratch](https://github.com/Windy3f3f3f3f/claude-code-from-scratch) 的 Python 实现。原始版权及衍生项目的许可见 [LICENSE](LICENSE)。
