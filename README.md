# Coding Agent Harness

面向软件工程任务的 Python Coding Agent 运行时。通过命令行串联代码探索、方案规划、工具执行、变更交付与测试验收。

- **Agent Loop 与多 Agent**。统一处理工具调用、权限检查与结果回写。支持 OpenAI 兼容和 Anthropic 兼容接口、Plan Mode、MCP，以及在独立上下文中执行任务的子 Agent。
- **测试验收**。根据 pytest 用例 ID 验证代码变更。用例失败时将诊断交回 Agent，可限制追加修复次数。跳过、未收集和超时会标记为未验证。
- **会话与 Skills**。支持恢复历史会话、折叠长对话、召回项目记忆，以及从用户反馈中抽取和评估可复用的技能。

## 本地运行

需要 Python 3.11 或更新版本。安装依赖后，将 `.env.example` 复制为 `.env`，填入自己的 API 密钥和模型配置。PowerShell 中可以使用以下命令。

```powershell
python -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements.txt
Copy-Item .env.example .env
& $py -m agents.main --help
& $py -m agents.main
```

在要处理的 Python 项目根目录运行 Agent，并把示例中的 pytest ID 换成实际用例。验收使用的解释器需要能导入 pytest 和目标项目依赖。可以用 `--test-python` 指定解释器。交互模式下可用 `/verify <用例 ID>` 单独验收。

```powershell
python -m agents.main --verify "tests/test_pricing.py::test_discount" "修复折扣边界问题"
```

项目也提供 Docker 构建文件和可复现的 [评测说明](eval/README.md)。

## 来源与许可

运行时基于 [Windy3f3f3f3f/claude-code-from-scratch](https://github.com/Windy3f3f3f3f/claude-code-from-scratch) 的 Python 实现。原始版权及衍生项目的许可见 [LICENSE](LICENSE)。
