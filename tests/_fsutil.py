"""試験の後片付け。Windows の git は .git の中のファイルを読み取り専用にするので、そのままでは shutil.rmtree で消せない
(2026-11-01 PC の試験で PermissionError と、Temp に .git の残ったフォルダが残る問題が出た)。読み取り専用を外してから消す。"""
import os, shutil, stat


def rmtree_force(path, ignore_errors=False):
    if not os.path.exists(path): return
    for dp, ds, fs in os.walk(path):
        for n in ds + fs:
            p = os.path.join(dp, n)
            try: os.chmod(p, stat.S_IWRITE | stat.S_IREAD | (stat.S_IEXEC if n in ds else 0))
            except OSError: pass
    shutil.rmtree(path, ignore_errors=ignore_errors)
