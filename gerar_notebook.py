"""Converte pipeline_cp3.py (células # %%) em notebook e executa todas as células."""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

fonte = Path("pipeline_cp3.py").read_text(encoding="utf-8")
links = dict(arg.split("=", 1) for arg in sys.argv[1:])
for k, v in links.items():
    fonte = fonte.replace(k, v)
celulas = []
for bloco in fonte.split("# %%")[1:]:
    if bloco.startswith(" [markdown]"):
        texto = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in bloco.split("\n")[1:]).strip()
        celulas.append(nbformat.v4.new_markdown_cell(texto))
    else:
        celulas.append(nbformat.v4.new_code_cell(bloco.strip("\n")))
nb = nbformat.v4.new_notebook(cells=celulas)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
saida = Path("CP03_DSA_Dashboard_MortalidadeTransitoSP_RM561942_RM562686.ipynb")
nbformat.write(nb, saida)
erros = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
print("celulas:", len(nb.cells), "| erros:", len(erros), "|", saida)
