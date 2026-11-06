"""Execute a project notebook using the Python environment running this script."""

import sys
import argparse
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", default="notebooks/00_data_audit.ipynb")
    parser.add_argument("--timeout", type=int, default=180, help="Maximum seconds per code cell; use 3600 for a cold full study.")
    args = parser.parse_args()
    path = (ROOT / args.notebook).resolve()
    if not path.is_relative_to(ROOT / "notebooks") or path.suffix != ".ipynb" or not path.is_file():
        parser.error("Select an existing .ipynb inside the project's notebooks folder.")
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    # Avoid accidentally using a globally registered kernel from another environment.
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=args.timeout, resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Executed {cells} code cells: {path}")


if __name__ == "__main__":
    main()
