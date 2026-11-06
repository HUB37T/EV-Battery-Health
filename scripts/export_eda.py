"""Export an executed EDA/preprocessing notebook as a reading copy without code."""

from pathlib import Path
import argparse

import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["eda", "preprocessing", "experiments"], default="eda")
    args = parser.parse_args()
    notebook_path,title,output = {
        "eda": ("notebooks/01_eda_train.ipynb", "EDA Mendalam EV Battery Health", "results/eda/EDA_READING.html"),
        "preprocessing": ("notebooks/02_preprocessing.ipynb", "Preprocessing EV Battery Health", "results/preprocessing/PREPROCESSING_READING.html"),
        "experiments": ("notebooks/03_mlp_experiments.ipynb", "Eksperimen MLP Backpropagation", "results/experiments/EXPERIMENT_READING.html"),
    }[args.stage]
    notebook = nbformat.read(ROOT / notebook_path, as_version=4)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    if not code or any(c.execution_count is None for c in code):
        raise RuntimeError("Execute the entire selected notebook before exporting.")
    if any(o.output_type == "error" for c in code for o in c.get("outputs", [])):
        raise RuntimeError("Notebook contains errors; fix before exporting.")
    exporter = HTMLExporter(template_name="lab", exclude_input=True, embed_images=True)
    body, _ = exporter.from_notebook_node(notebook, resources={"metadata": {"name": title}})
    body = body.replace('<html lang="en">', '<html lang="id">')
    body = "\n".join(line.rstrip() for line in body.splitlines()).rstrip() + "\n"
    path = ROOT / output
    path.write_text(body, encoding="utf-8", newline="\n")
    print(path)


if __name__ == "__main__":
    main()
