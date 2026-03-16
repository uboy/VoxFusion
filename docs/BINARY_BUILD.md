# Binary Build and Packaging

VoxFusion produces two binaries:

- `voxfusion-gui` for graphical mode.
- `voxfusion-cli` for command-line mode.

Build script: `scripts/build_binaries.py`

## Why `--onedir`

For all platforms we use PyInstaller `--onedir`.
For Windows this is mandatory to reduce Smart App Control blocking risk and keep DLLs next to `.exe`.

## Build commands

```bash
python scripts/build_binaries.py --target all
python scripts/build_binaries.py --target gui
python scripts/build_binaries.py --target cli
```

Output:

- Bundle directories: `dist/binaries/voxfusion-gui`, `dist/binaries/voxfusion-cli`
- ZIP archives: `dist/binaries/voxfusion-gui-<platform>.zip`, `dist/binaries/voxfusion-cli-<platform>.zip`

## Windows specifics

1. Build stays in `--onedir` mode (no single-file self-extracting EXE).
2. A PyInstaller `version-file` is generated with current classes:
   - `FixedFileInfo`
   - `StringStruct`
   - `VarStruct`
3. Users should unblock the downloaded ZIP in file properties before extraction.

## Included data

Build script explicitly adds:

- `src/voxfusion/config/defaults.yaml`
- `customtkinter/assets/themes` (if `customtkinter` is installed)

## Launchers

- GUI launcher: `gui_start.py`
- CLI launcher: `cli_start.py`

Both launchers use absolute package imports only.
