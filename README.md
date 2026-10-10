# Data Science Bootcamp - Lecture 1

Build a claim-cost model for auto insurance, end to end, on synthetic data.

- Start with the [class project intro](data_science_bootcamp_lecture_1_class_project_intro.md) (business problem and data).
- Then run the four notebooks in `analysis_pipeline/` in order: `1_data_etl` -> `2_data_exploration` -> `3_model_selection` -> `4_model_deployment`.

## Setup

### 1. Install VS Code

- [Download VS Code](https://code.visualstudio.com/)
- Install the **Python** and **Jupyter** extensions (Extensions icon in the left bar, search each name).

### 2. Clone and Open the Repo

In the VS Code terminal, run:

```
git clone https://github.com/mengjin67/data_science_bootcamp_lecture_1.git
```

Then open the cloned folder (File > Open Folder).

### 3. Install `uv`

- **macOS/Linux:**

  ```
  sudo chown -R $USER ~/.config 2>/dev/null; mkdir -p ~/.config/fish/conf.d 2>/dev/null; UV_NO_MODIFY_PATH=1 curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Windows:**

  ```
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

- Check the install (open a new terminal first): `uv --version`
- Stuck? See [Manual install](#uv-manual-binary-installation) below or ask AI.

### 4. Create the Environment

In the VS Code terminal, from the repo root, run:

```
uv sync
```

This creates the `.venv` folder (Python 3.10) and installs all packages. The first run takes several minutes (large packages such as `torch`).

### 5. Select the Kernel

A notebook only runs with the packages from `.venv` if you select it as the kernel. Do this for **each notebook**, the first time you open it:

1. Open a notebook, e.g. `analysis_pipeline/1_data_etl.ipynb`.
2. Click **Select Kernel** in the top-right corner of the notebook.
3. Choose **Python Environments...**, then the entry that says `.venv` (Python 3.10.x). Its path ends in `data_science_bootcamp_lecture_1/.venv/bin/python` (macOS/Linux) or `data_science_bootcamp_lecture_1\.venv\Scripts\python.exe` (Windows).
4. Check the top-right corner now shows `.venv (Python 3.10.x)`.

Not in the list?

- Make sure step 4 finished (a `.venv` folder exists in the repo root), then reload VS Code: `Cmd+Shift+P` (Windows: `Ctrl+Shift+P`) > **Developer: Reload Window**.
- Still missing: **Select Kernel** > **Select Another Kernel...** > **Python Environments...** > **Enter interpreter path...** > **Find...**, and pick the `python` file path from above.
- Do **not** pick Anaconda or the system Python: they lack this project's packages.

### 6. Run the Notebooks

1. In `1_data_etl.ipynb`, click **Run All**. It creates the `.local` folder and downloads the data (internet required).
2. Run `2_data_exploration.ipynb`, `3_model_selection.ipynb` and `4_model_deployment.ipynb` in order. Each reads the previous notebook's output from `.local`.

If all four run without errors, your setup is complete.

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` | Open a new terminal; if still missing, see the manual install below |
| `ModuleNotFoundError` (e.g. `No module named 'xgboost'`) | Wrong kernel: redo step 5 and pick the `.venv` kernel, then restart the kernel |
| `.venv` kernel not listed | See "Not in the list?" in step 5 |
| `FileNotFoundError` on a `.local/...parquet` file | Run the earlier notebooks first (1 -> 4 in order) |
| Data download fails | Check your internet connection and re-run the download cell in notebook 1 |

---

## uv Manual Binary Installation

Download the latest asset from [uv releases](https://github.com/astral-sh/uv/releases).

### Windows

1. Download: `uv-x86_64-pc-windows-msvc.zip` (Intel/AMD) or `uv-aarch64-pc-windows-msvc.zip` (ARM).
2. Unzip to get `uv.exe`.
3. Move it to a folder such as `C:\Program Files\uv\`.
4. Add that folder to PATH: System Properties > Advanced > Environment Variables > System variables > Path > Edit > New.
5. Restart your terminal/VS Code.

### macOS

1. Download: `uv-aarch64-apple-darwin.tar.gz` (Apple Silicon) or `uv-x86_64-apple-darwin.tar.gz` (Intel).
2. Double-click the file to extract; you get a single file named `uv`.
3. Move `uv` to `/usr/local/bin` (all Macs) or `/opt/homebrew/bin` (Homebrew on Apple Silicon). Enter your admin password if asked.
4. Make it executable:

   ```
   chmod +x /usr/local/bin/uv
   ```

   (use `/opt/homebrew/bin/uv` if you placed it there)

### If `uv` is still not found

- Windows: recheck the PATH entry and reopen the terminal.
- macOS: `echo $PATH` should include the folder you chose.
