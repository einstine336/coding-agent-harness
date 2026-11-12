"""Smoke tests for the existing Harness modules using only isolated synthetic data."""

from contextlib import chdir
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from agents import memory, session, skills
from agents.agent import Agent
from agents.mcp_client import McpManager
from agents.online_skill_eval import _build_replay_pool, _evaluate_rule
from agents.online_skill_evolution import online_ingest
from agents.tools import check_permission


def stub_agent(**kwargs):
    return Agent(api_base="http://127.0.0.1:1/v1", model="test-stub", api_key="offline",
                 is_sub_agent=True, custom_system_prompt="offline test", **kwargs)


MCP_SERVER = """
import json
import sys
for line in sys.stdin:
    request = json.loads(line)
    method = request["method"]
    if "id" not in request:
        continue
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "serverInfo": {"name": "local", "version": "1"},
                  "capabilities": {"tools": {}}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": request["params"]["arguments"]["text"]}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


class HarnessModuleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = self.enterContext(tempfile.TemporaryDirectory(prefix="bear-modules-"))
        self.root = Path(root)
        self.enterContext(chdir(self.root))
        self.enterContext(patch("pathlib.Path.home", return_value=self.root))
        self.enterContext(patch.object(session, "SESSION_DIR", self.root / "sessions"))
        skills.reset_skill_cache()
        self.addCleanup(skills.reset_skill_cache)

    async def test_session_fold_persist_and_resume(self):
        agent = stub_agent()
        self.addAsyncCleanup(agent.shutdown)
        agent._openai_messages.extend([
            {"role": "user", "content": "Fix the billing bug"},
            {"role": "assistant", "content": "I found the cause"},
            {"role": "user", "content": "Keep discounts unchanged"},
        ])
        structured = {
            "episode_memory": {"current_progress": "cause found"},
            "working_memory": {"next_actions": [{"type": "edit", "description": "fix only billing"}]},
            "tool_memory": {"derived_rules": ["preserve discount regression"]},
        }

        async def side_query(system, user):
            self.assertIn("Keep discounts unchanged", user)
            return json.dumps(structured)

        with patch.object(agent, "_build_side_query", return_value=side_query):
            self.assertTrue(await agent._compact_conversation(trigger="smoke"))
        self.assertEqual(len(agent._openai_messages), 2)
        folded = agent._openai_messages[-1]["content"]
        for evidence in ("cause found", "fix only billing", "preserve discount regression"):
            self.assertIn(evidence, folded)
        agent._auto_save()
        restored = stub_agent()
        self.addAsyncCleanup(restored.shutdown)
        restored.restore_session(session.load_session(agent.session_id))
        self.assertEqual(restored._openai_messages[-1]["content"], folded)
        self.assertEqual(restored._folded_session_memories[-1]["trigger"], "smoke")
        self.assertTrue(list((self.root / ".bear" / "sessions").glob("*.folded-memory.jsonl")))

    async def test_memory_recall_excludes_already_injected(self):
        filename = memory.save_memory("billing-rules", "billing policy", "project", "Always preserve fractional discounts")
        calls = []

        async def side_query(system, user):
            calls.append(user)
            return json.dumps({"selected_memories": [filename]})

        selected = await memory.select_relevant_memories("discounts", side_query, set())
        self.assertEqual(len(selected), 1)
        self.assertIn("fractional discounts", selected[0].content)
        self.assertEqual(await memory.select_relevant_memories("discounts", side_query, {selected[0].path}), [])
        self.assertEqual(len(calls), 1)

    async def test_skill_evolution_add_merge_discard_and_retrieval(self):
        candidate = {
            "name": "billing-regression-check", "description": "Check billing regression",
            "when_to_use": "When changing billing discounts", "instructions": "Run the billing tests.",
            "evidence": "User requests a regression check.", "tags": ["billing"],
        }

        async def ingest(action, item, instruction=""):
            async def side_query(system, user):
                if "Skill Extractor" in system:
                    return json.dumps({"skills": [item]})
                return json.dumps({"action": action, "target_skill": candidate["name"],
                                   "merged_instructions": instruction})
            return await online_ingest(messages=[{"role": "user", "content": item["evidence"]}], side_query=side_query)

        added = await ingest("add", candidate)
        self.assertTrue(added["ok"], added)
        self.assertEqual(added["action"], "add")
        skill_path = self.root / ".bear" / "skills" / candidate["name"] / "SKILL.md"
        self.assertIn("Run the billing tests.", skill_path.read_text(encoding="utf-8"))
        merged = await ingest("merge", candidate, "Run both billing discount and refund tests.")
        self.assertTrue(merged["ok"], merged)
        self.assertEqual(merged["action"], "merge")
        self.assertIn("refund tests", skill_path.read_text(encoding="utf-8"))
        discarded = await ingest("discard", dict(candidate, name="unrelated", description="misc",
                                                 when_to_use="misc", evidence="Temporary style request."))
        self.assertEqual(discarded["action"], "discard")
        self.assertEqual(len(list((self.root / ".bear" / "skills").glob("*/SKILL.md"))), 1)
        hits = skills.retrieve_relevant_skills("billing discounts regression", limit=1)
        self.assertEqual(hits[0]["name"], candidate["name"])
        self.assertTrue(list((self.root / ".bear" / "skill-evolution").glob("online_provenance.jsonl")))

    async def test_skill_replay_deduplicates_and_rule_distinguishes_outputs(self):
        rows = [{"messages": [{"role": "user", "content": f"Return JSON for input {i}"}]} for i in range(5)]
        pool = _build_replay_pool("json-output", rows + rows[:1], {}, freeze=False)
        self.assertEqual(len(pool), 5)
        self.assertEqual({row["split"] for row in pool}, {"mutate_dev", "promotion_test"})
        rule = {"rule_id": "json", "params": {"mode": "json_parseable"}}
        self.assertTrue(_evaluate_rule(rule, '{"ok": true}')["passed"])
        self.assertFalse(_evaluate_rule(rule, '{"ok":}')["passed"])

    async def test_plan_mode_blocks_shell_and_edits(self):
        self.assertEqual(check_permission("read_file", {"file_path": "fixture.py"}, "plan")["action"], "allow")
        self.assertEqual(check_permission("run_shell", {"command": "pwd"}, "plan")["action"], "deny")
        self.assertEqual(check_permission("edit_file", {"file_path": "fixture.py"}, "plan")["action"], "deny")

    async def test_subagents_share_provider_and_permission_but_have_own_history(self):
        parent = Agent(anthropic_base_url="http://127.0.0.1:1/anthropic", model="test-stub", api_key="offline",
                       custom_system_prompt="offline test", is_sub_agent=True, permission_mode="dontAsk")
        self.addAsyncCleanup(parent.shutdown)
        parent._anthropic_messages.append({"role": "user", "content": "parent only"})
        visited = []

        async def run_once(child, prompt):
            self.assertTrue(child.is_sub_agent)
            self.assertEqual(child.permission_mode, "dontAsk")
            self.assertEqual(str(child._anthropic_client.base_url), str(parent._anthropic_client.base_url))
            self.assertEqual(child._anthropic_messages, [])
            visited.append(prompt)
            return {"text": f"done: {prompt}", "tokens": {"input": 2, "output": 1}}

        with patch.object(Agent, "run_once", run_once):
            for role in ("explore", "plan", "general"):
                result = await parent._execute_agent_tool({"type": role, "prompt": role})
                self.assertEqual(result, f"done: {role}")
        self.assertEqual(visited, ["explore", "plan", "general"])
        self.assertEqual(parent.get_token_usage(), {"input": 6, "output": 3})

    async def test_mcp_stdio_handshake_discovery_call_and_close(self):
        script = self.root / "fixture_mcp.py"
        script.write_text(MCP_SERVER, encoding="utf-8")
        manager = McpManager()
        self.addAsyncCleanup(manager.disconnect_all)
        with patch.object(manager, "_load_configs", return_value={
            "local": {"command": sys.executable, "args": ["-B", str(script)]},
        }):
            await manager.load_and_connect()
        self.assertEqual([tool["name"] for tool in manager.get_tool_definitions()], ["mcp__local__echo"])
        self.assertEqual(await manager.call_tool("mcp__local__echo", {"text": "fixture"}), "fixture")
        await manager.disconnect_all()
        self.assertEqual(manager.get_tool_definitions(), [])


if __name__ == "__main__":
    unittest.main()
