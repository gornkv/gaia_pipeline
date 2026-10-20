"""Проверки общей библиотеки фич после упрощения структуры scaffold."""

import importlib
import os
import pkgutil
import unittest
from unittest.mock import patch

import svc_scaffold.features as feature_modules
from svc_scaffold.features import common
from svc_scaffold.core import enabled_features


class ScaffoldTest(unittest.TestCase):
    def test_all_features_import(self):
        # Динамическое обнаружение — единственная связь core с отдельными файлами фич.
        names = []
        with patch.dict(os.environ, {"MODEL_CONTEXT_LIMIT": "32768"}):
            for module in pkgutil.iter_modules(
                feature_modules.__path__, feature_modules.__name__ + "."
            ):
                if module.name.endswith(".common") or module.name.rsplit(".", 1)[-1].startswith("-"):
                    continue
                names.append(importlib.import_module(module.name).Breakpoints.feature_name())
        self.assertIn("budget_tracker", names)
        self.assertEqual(len(names), len(set(names)))

    def test_common_module_is_not_loaded_as_feature(self):
        with patch.dict(os.environ, {"FEATURES": "budget_tracker", "MODEL_CONTEXT_LIMIT": "32768"}):
            handlers = enabled_features()
        self.assertEqual([handler.feature_name() for handler in handlers], ["budget_tracker"])

    def test_csv_feature_order_is_handler_order(self):
        with patch.dict(os.environ, {"FEATURES": "voi,budget_tracker", "MODEL_CONTEXT_LIMIT": "32768"}):
            handlers = enabled_features()
        self.assertEqual([handler.feature_name() for handler in handlers], ["voi", "budget_tracker"])

    def test_branch_context_keeps_only_current_branch(self):
        context = common.BranchContextManager()
        context.start({"messages": [{"role": "user", "content": "task"}]})
        context.after_chat_message({
            "choices": [{"message": {
                "role": "assistant", "tool_calls": [{"id": "new"}],
            }}]
        })
        context.after_tool_call({"messages": [
            {"role": "tool", "tool_call_id": "old", "content": "old"},
            {"role": "tool", "tool_call_id": "new", "content": "new"},
        ]})
        isolated = context.before_chat_message({"messages": [{"role": "user", "content": "noise"}]})
        self.assertEqual([item.get("content") for item in isolated["messages"]], ["task", None, "new"])


if __name__ == "__main__":
    unittest.main()
