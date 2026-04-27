"""Override PyInstaller's torch hook to avoid segfault in collect_submodules.

The upstream hook calls collect_submodules('torch') which triggers
_collect_submodules('torch.distributed.optim') in an isolated subprocess
that segfaults on Linux.  We replace it with a safe version that only
collects data files and dynamic libs without recursing into submodules.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files("torch")
binaries = collect_dynamic_libs("torch")
