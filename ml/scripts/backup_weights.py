"""Резервная копия весов YOLO перед переобучением.

Использование:
    python3 ml/scripts/backup_weights.py pricetag_v2
    # → копирует ml/output/runs/pricetag_v2/ в
    #   ml/output/runs/_backup/pricetag_v2__<timestamp>/

Вызывается автоматически из train_yolo.py перед запуском тренировки —
если name совпадает с уже существующим run'ом, его кладут в _backup/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "ml" / "output" / "runs"
BACKUP_DIR = RUNS_DIR / "_backup"


def backup_run(name: str) -> Path | None:
    src = RUNS_DIR / name
    if not src.exists():
        print(f"  no run to backup: {src}")
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"{name}__{ts}"
    # копируем всё содержимое run'а (включая weights/, args.yaml, плоты)
    shutil.copytree(src, dst)
    print(f"  backed up: {src} -> {dst}")
    return dst


def list_backups(name: str | None = None):
    if not BACKUP_DIR.exists():
        print("  (empty)")
        return
    items = sorted(BACKUP_DIR.iterdir(), key=lambda p: p.name, reverse=True)
    if name:
        items = [p for p in items if p.name.startswith(f"{name}__")]
    for p in items:
        try:
            sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            sz_mb = sz / 1024 / 1024
        except Exception:
            sz_mb = 0
        print(f"  {p.name}  ({sz_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    p_bk = sub.add_parser("backup", help="бэкап указанного run'а")
    p_bk.add_argument("name", help="имя run'а (напр. pricetag_v2)")

    p_ls = sub.add_parser("list", help="перечислить бэкапы")
    p_ls.add_argument("--name", default=None)

    p_rs = sub.add_parser("restore",
                          help="восстановить бэкап в основное место")
    p_rs.add_argument("backup_name", help="имя из _backup/")

    args = ap.parse_args()
    if args.cmd == "backup":
        backup_run(args.name)
    elif args.cmd == "list":
        list_backups(args.name)
    elif args.cmd == "restore":
        src = BACKUP_DIR / args.backup_name
        if not src.exists():
            print(f"  not found: {src}", file=sys.stderr)
            return 1
        # имя оригинала — всё до последнего "__<timestamp>"
        orig = args.backup_name.rsplit("__", 1)[0]
        dst = RUNS_DIR / orig
        if dst.exists():
            # делаем встречный бэкап на всякий случай
            print(f"  warn: {dst} уже существует — делаю встречный бэкап")
            backup_run(orig)
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  restored: {src} -> {dst}")
    else:
        ap.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
