"""Execute the audit notebook using the Python environment running this script."""

import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/00_data_audit.ipynb"


def main():
    notebook = nbformat.read(PATH, as_version=4)
    manager = KernelManager(kernel_name="python3")
    # Avoid accidentally using a globally registered kernel from another environment.
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, PATH)
    cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Executed {cells} code cells: {PATH}")


if __name__ == "__main__":
    main()
