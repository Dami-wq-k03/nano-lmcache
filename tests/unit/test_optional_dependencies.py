import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

CONFTEST_PATH = Path(__file__).resolve().parents[1] / "conftest.py"


def load_conftest() -> ModuleType:
    """Load tests/conftest.py by path to test its skip logic directly."""
    spec = importlib.util.spec_from_file_location("nano_conftest", CONFTEST_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_the_package_does_not_import_torch():
    # Run in a fresh interpreter so other tests cannot have imported torch.
    code = (
        "import sys\n"
        "import nano_lmcache\n"
        "import nano_lmcache.core.chunking\n"
        "import nano_lmcache.core.metrics\n"
        "import nano_lmcache.core.prefix_cache\n"
        "import nano_lmcache.storage.memory\n"
        "import nano_lmcache.storage.protocol\n"
        "heavy = [m for m in ('torch', 'transformers', 'vllm') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_unmarked_tests_are_never_skipped():
    conftest = load_conftest()

    reason = conftest.skip_reason(
        set(), has_torch=False, has_cuda=False, has_vllm=False
    )

    assert reason is None


def test_torch_marker_needs_torch():
    conftest = load_conftest()

    assert conftest.skip_reason(
        {"torch"}, has_torch=False, has_cuda=False, has_vllm=False
    )
    assert (
        conftest.skip_reason(
            {"torch"}, has_torch=True, has_cuda=False, has_vllm=False
        )
        is None
    )


def test_gpu_marker_needs_torch_and_cuda():
    conftest = load_conftest()

    assert conftest.skip_reason(
        {"gpu"}, has_torch=False, has_cuda=False, has_vllm=False
    )
    # torch installed but no GPU: skipped, not failed.
    assert conftest.skip_reason(
        {"gpu"}, has_torch=True, has_cuda=False, has_vllm=False
    )
    assert (
        conftest.skip_reason(
            {"gpu"}, has_torch=True, has_cuda=True, has_vllm=False
        )
        is None
    )


def test_vllm_marker_needs_vllm_and_cuda():
    conftest = load_conftest()

    assert conftest.skip_reason(
        {"vllm"}, has_torch=True, has_cuda=True, has_vllm=False
    )
    assert conftest.skip_reason(
        {"vllm"}, has_torch=True, has_cuda=False, has_vllm=True
    )
    assert (
        conftest.skip_reason(
            {"vllm"}, has_torch=True, has_cuda=True, has_vllm=True
        )
        is None
    )


def test_skip_reason_names_what_is_missing():
    conftest = load_conftest()

    reason = conftest.skip_reason(
        {"gpu"}, has_torch=False, has_cuda=False, has_vllm=False
    )

    assert "torch" in reason


def test_markers_are_registered():
    text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()

    for marker in ("torch:", "gpu:", "vllm:"):
        assert f'"{marker}' in text


def test_core_package_declares_no_runtime_dependencies():
    text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()

    assert "dependencies = []" in text
    assert importlib.util.find_spec("nano_lmcache") is not None
