"""Export an executed EDA notebook as a standalone reading copy, without code."""

from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook = nbformat.read(ROOT / "notebooks/01_eda_train.ipynb", as_version=4)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    if not code or any(c.execution_count is None for c in code):
        raise RuntimeError("Execute the entire EDA notebook before exporting.")
    if any(o.output_type == "error" for c in code for o in c.get("outputs", [])):
        raise RuntimeError("Notebook contains errors; fix before exporting.")
    exporter = HTMLExporter(template_name="lab", exclude_input=True, embed_images=True)
    body, _ = exporter.from_notebook_node(notebook, resources={"metadata": {"name": "EDA Mendalam EV Battery Health"}})
    body = body.replace('<html lang="en">', '<html lang="id">')
    body = "\n".join(line.rstrip() for line in body.splitlines()).rstrip() + "\n"
    path = ROOT / "results/eda/EDA_READING.html"
    path.write_text(body, encoding="utf-8", newline="\n")
    print(path)


if __name__ == "__main__":
    main()
