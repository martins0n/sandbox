import importlib.util
import glob
import pathlib

import subprocess




def apply_patch(library_name: str):
    spec = pathlib.Path(importlib.util.find_spec(library_name).origin).parent
    for patch in pathlib.Path(__file__).parent.glob("**/*.patch"):
        print(f"Applying {patch}")
        relative_path = patch.relative_to(pathlib.Path(__file__).parent)
        
        dir_with_file_to_patch = spec.parent / relative_path.parent
        file_to_patch = patch.name.replace(".patch", ".py")
        
        full_path = dir_with_file_to_patch / file_to_patch
        
        assert full_path.exists(), f"File {full_path} does not exist"
        
        subprocess.run(["patch", "-i", str(patch), str(full_path)])        

apply_patch("transformers")