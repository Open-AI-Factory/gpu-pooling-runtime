import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_operator_imports_without_torch():
    code = ("import sys; sys.modules['torch'] = None; sys.path.insert(0, sys.argv[1]); "
            "import grp_runtime.runtime_coordination.runtime_action_controller.operator")
    out = subprocess.run([sys.executable, "-c", code, str(SRC)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]


def test_hook_still_exported_for_workloads():
    from grp_runtime.agent import RuntimeHook
    assert RuntimeHook.__name__ == "RuntimeHook"
