import importlib.util
from pathlib import Path
import subprocess
import unittest


class VllmCumemRuntimeTests(unittest.TestCase):
    def test_cumem_extension_uses_runtime_safe_none_reference(self):
        vllm_spec = importlib.util.find_spec("vllm")
        if vllm_spec is None or not vllm_spec.submodule_search_locations:
            self.skipTest("vLLM is not installed")

        package_dir = Path(next(iter(vllm_spec.submodule_search_locations)))
        extensions = list(package_dir.glob("cumem_allocator*.so"))
        self.assertEqual(len(extensions), 1, extensions)

        symbols = subprocess.run(
            ["nm", "-D", str(extensions[0])],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        self.assertIn(
            "Py_IncRef",
            symbols,
            "unsafe vLLM abi3 cumem extension: None is returned without a "
            "runtime reference increment on CPython 3.10/3.11",
        )


if __name__ == "__main__":
    unittest.main()
