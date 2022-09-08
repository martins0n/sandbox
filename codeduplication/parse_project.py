import ast
from collections import defaultdict
import intervaltree
import pathlib
import subprocess
import os
import shutil
import json

import argparse


GIT_PROJECT = "https://github.com/tinkoff-ai/etna"


def parse_file_to_intervaltree(file_path: str) -> intervaltree.IntervalTree:
    """https://julien.danjou.info/finding-definitions-from-a-source-file-and-a-line-number-in-python"""

    def node_interval(node: ast.stmt):
        min_ = node.lineno
        max_ = node.lineno
        for node in ast.walk(node):
            if hasattr(node, "lineno"):
                min_ = min(min_, node.lineno)
                max_ = max(max_, node.lineno)
        return min_, max_ + 1

    with open(file_path, "r") as f:
        file_content = f.read()
        parsed = ast.parse(file_content)
    file_content = file_content.splitlines()
    tree = intervaltree.IntervalTree()
    for item in ast.walk(parsed):
        if isinstance(item, (ast.ClassDef, ast.FunctionDef)):
            interval_ = node_interval(item)
            tree[interval_[0] : interval_[1]] = dict(
                name=item.name,
                source_code="\n".join(file_content[interval_[0] - 1 : interval_[1]]),
            )
            break

    return tree


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--project", type=str, default=GIT_PROJECT)
    
    args = parser.parse_args()
    
    git_project = args.project

    FILE_DIR = pathlib.Path(__file__).parent
    TMP_DIR = FILE_DIR / "tmp"
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    TMP_DIR.mkdir(exist_ok=True)

    os.chdir(TMP_DIR)

    subprocess.run(["git", "clone", "--depth=1", git_project, "."])

    index = defaultdict(dict)
    for path in pathlib.Path(".").glob("**/*.py"):
        tree = parse_file_to_intervaltree(str(path))
        index[str(path)] = defaultdict(dict)
        _index = index[str(path)]
        for interval in tree:
            _index[interval.data["name"]] = dict(
                source_code=interval.data["source_code"],
                lines=[interval[0], interval[1]],
            )

    with open("index.json", "w") as f:
        json.dump(index, f, indent=4)
