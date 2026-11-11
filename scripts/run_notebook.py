"""Execute project notebooks in place with the Python environment running this script.

python scripts/run_notebook.py notebooks/01_eda.ipynb
python scripts/run_notebook.py --all          # 00 -> 05 in order
"""

import argparse
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def execute(path: Path, timeout: int) -> None:
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    # Avoid accidentally using a globally registered kernel from another environment.
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(
        notebook, km=manager, timeout=timeout, resources={"metadata": {"path": str(ROOT)}}
    )
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Executed {cells} code cells: {path.relative_to(ROOT)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("notebook", nargs="?", help="path to one notebook inside notebooks/")
    parser.add_argument("--all", action="store_true", help="execute every notebook in numeric order")
    parser.add_argument("--timeout", type=int, default=3600, help="maximum seconds per code cell")
    args = parser.parse_args()
    if args.all == bool(args.notebook):
        parser.error("Give exactly one notebook path or --all.")
    if args.all:
        paths = sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))
    else:
        path = (ROOT / args.notebook).resolve()
        if not path.is_relative_to(NOTEBOOKS) or path.suffix != ".ipynb" or not path.is_file():
            parser.error("Select an existing .ipynb inside the project's notebooks folder.")
        paths = [path]
    for path in paths:
        execute(path, args.timeout)


if __name__ == "__main__":
    main()
