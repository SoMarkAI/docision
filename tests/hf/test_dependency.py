import ast
from pathlib import Path

def test_hf_path_never_calls_generate_or_imports_vllm():
    root=Path(__file__).resolve().parents[2]
    for name in ['docision/_runtime/hf.py','docision/_runtime/prompt.py','demo_docision.py']:
        tree=ast.parse((root/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr!='generate'
            if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('vllm')
            if isinstance(node,ast.Import):assert all(not n.name.startswith('vllm') for n in node.names)
