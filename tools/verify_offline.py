"""离线验证 Bear Agent 的完整 tool loop，不需要任何真实 API Key。

本地起一个 OpenAI-compatible 的假模型服务，让 Agent 真的走一遍：
  模型请求 -> tool_call -> 权限判断 -> 工具执行 -> 结果回写 -> 再次请求 -> 结束

用法（在仓库根目录）：
  python tools/verify_offline.py
  python tools/verify_offline.py --python ../.venv-bear/Scripts/python.exe
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "mock-model"


def chunk(cid, delta, finish_reason=None, usage=None):
    body = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        body["choices"] = []
        body["usage"] = usage
    return body


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception:
            req = {}

        server = self.server
        if not req.get("stream"):
            # side query：记忆召回 / 技能检索 / 后台抽取，返回空结果让流程安全通过
            server.side_calls += 1
            body = {
                "id": f"side-{server.side_calls}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "[]"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
            }
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        server.main_calls += 1
        turn = server.main_calls
        cid = f"main-{turn}"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(obj):
            try:
                self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
                self.wfile.write(b"")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        if turn == 1:
            # 第一轮：模型提出工具调用，读 agents 目录下的 Python 模块
            send(chunk(cid, {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "index": 0,
                    "id": "call_list_1",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": ""},
                }],
            }))
            args = json.dumps({"pattern": "agents/*.py"}, ensure_ascii=False)
            send(chunk(cid, {"tool_calls": [{
                "index": 0,
                "function": {"arguments": args},
            }]}))
            send(chunk(cid, {}, finish_reason="tool_calls"))
            send(chunk(cid, {}, usage={
                "prompt_tokens": 120, "completion_tokens": 18, "total_tokens": 138,
            }))
        else:
            # 第二轮：拿到工具结果，给出自然语言结论并结束循环
            text = "已列出 agents 目录下的 Python 模块，共 15 个，工具链验证完成。"
            for piece in [text[:12], text[12:]]:
                send(chunk(cid, {"content": piece}))
            send(chunk(cid, {}, finish_reason="stop"))
            send(chunk(cid, {}, usage={
                "prompt_tokens": 260, "completion_tokens": 24, "total_tokens": 284,
            }))

        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True


def main():
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(tools_dir))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--prompt", default="列出 agents 目录下的 Python 模块")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    if not os.path.isdir(os.path.join(repo, "agents")):
        print(f"找不到 agents/ 目录：{repo}")
        return 2

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.main_calls = 0
    server.side_calls = 0
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[mock] OpenAI-compatible 假模型已启动：http://127.0.0.1:{port}/v1")

    env = os.environ.copy()
    env.update({
        "OPENAI_API_KEY": "sk-offline-mock",
        "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "MODEL": MODEL,
        "NO_COLOR": "1",
        "TERM": "dumb",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    })
    # 避免 .env 里残留的 DeepSeek 配置覆盖假模型地址
    for k in ("APIKEY", "API", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
        env.pop(k, None)

    cmd = [
        args.python, "-m", "agents.main",
        "--dont-ask", "--max-turns", "5", "--model", MODEL,
        args.prompt,
    ]
    print(f"[run] cwd={repo}")
    print(f"[run] {' '.join(cmd)}\n" + "-" * 60)
    try:
        p = subprocess.run(
            cmd, cwd=repo, env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=180,
        )
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc = -1
    finally:
        server.shutdown()
        server.server_close()

    print(out)
    if err.strip():
        print("-" * 60 + "\n[stderr]\n" + err[-2000:])

    print("-" * 60)
    print(f"[mock] 主循环请求 {server.main_calls} 次，side query {server.side_calls} 次")
    checks = {
        "CLI 正常退出": rc == 0,
        "模型主循环跑满两轮": server.main_calls >= 2,
        "工具 list_files 被调用": "list_files" in out,
        "读到真实文件": ("agent.py" in out or "agents/" in out),
        "没有 Python 异常": "Traceback (most recent call last)" not in (out + err),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    ok = all(checks.values()) and rc == 0
    print(f"\n结论：{'整条 tool loop 离线跑通' if ok else '未完全通过，看上面的输出'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
