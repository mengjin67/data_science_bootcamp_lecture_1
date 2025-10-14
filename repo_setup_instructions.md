# Repo Setup Instructions

## 1. Download and Install VS Code

- [VS Code Download](https://code.visualstudio.com/)

## 2. Clone the GitHub Repository

Open VS Code and run the following command in the terminal to clone the repo to a local folder:

```

git clone https://github.com/mengjin67/data_science_bootcamp_lecture_1.git

```

## 3. Open the Cloned Repo

- Use the VS Code file explorer to open the cloned repository folder.

## 4. Install `uv`

-**MacOS/Linux:**

```

  sudo chown -R $USER ~/.config 2>/dev/null; mkdir -p ~/.config/fish/conf.d 2>/dev/null; UV_NO_MODIFY_PATH=1 curl -LsSf https://astral.sh/uv/install.sh | sh

```

-**Windows:**

```

  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

```

- If you have trouble installing `uv` or have any questions, ask AI.
- Check if `uv` is successfully installed:

  ```

  uv --version

  ```

## 5. Set Up the Project Folder

Open a terminal in VS Code and run:

```

uv venv

uv sync

```

## 6. Add Extensions in VS Code

- Install the **Python** extension.
- Install the **Jupyter** extension.

## 7. Create `.local` Folder

- Create a `.local` folder under your repository.

## 8. Run the Notebook

- Open and run `1_data_etl.ipynb` under the `analysis_pipeline` folder.
- If it runs successfully, your setup is complete!

---

## uv Manual Binary Installation

### Windows (manual .zip install)

1.**Download**

- Go to: [uv releases](https://github.com/astral-sh/uv/releases)
- Download the latest Windows asset:

  - Intel/AMD: `uv-x86_64-pc-windows-msvc.zip`
  - ARM (Surface/ARM PCs): `uv-aarch64-pc-windows-msvc.zip`

    2.**Extract**

- Unzip; you’ll get `uv.exe`.

  3.**Place the Binary**

- Create a folder, e.g. `C:\Program Files\uv\`
- Move `uv.exe` there.

  4.**Add to PATH**

- Open System Properties → Advanced → Environment Variables
- Under System variables → select Path → Edit → New
- Add: `C:\Program Files\uv\`
- OK all dialogs, then restart your apps/terminal.

### macOS (manual .tar.gz install)

1.**Download**

- Go to: [uv releases](https://github.com/astral-sh/uv/releases)
- Download the latest macOS asset:

  - Apple Silicon (M1/M2/M3): `uv-aarch64-apple-darwin.tar.gz`
  - Intel: `uv-x86_64-apple-darwin.tar.gz`

    2.**Extract**

- Double-click the `.tar.gz`; you’ll get a folder containing a single file named `uv`.

  3.**Place the Binary**

- Move `uv` to one of:

  -`/usr/local/bin` (works on all Macs), or

  -`/opt/homebrew/bin` (common on Apple Silicon with Homebrew)

  4.**Make Executable (if needed)**

```

   chmod +x /usr/local/bin/uv

   # or, if placed in Homebrew bin:

   chmod +x /opt/homebrew/bin/uv

```

### Notes / Troubleshooting

- If `uv` isn’t found, ensure the folder you chose is on your PATH:

  -**Windows:** Recheck the Path entry and reopen your terminal/IDE.

  -**macOS:**`echo $PATH` should include `/usr/local/bin` or `/opt/homebrew/bin`.

- If macOS blocks the move, authenticate when prompted (admin password).
