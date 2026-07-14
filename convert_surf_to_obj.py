#!/usr/bin/env python3
"""
Convert FreeSurfer .surf files to .obj files for Blender.

Run with:
    python convert_surf_to_obj.py
"""

from pathlib import Path
import mne

# Change subject code 
input_dir = Path("/work/uphummel/studies/tTIS-EEG/derivatives/MRI/freesurfer/sub-41Y01_ses-baseline/bem")

output_dir = Path("/work/uphummel/studies/tTIS-EEG/derivatives/MRI/freesurfer/sub-41Y01_ses-baseline/conv")

surfaces = ["brain.surf", "inner_skull.surf", "outer_skull.surf", "outer_skin.surf"]

overwrite = True

def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input folder:  {input_dir}")
    print(f"Output folder: {output_dir}")

    for surface_name in surfaces:
        surf_path = input_dir / surface_name
        obj_path = output_dir / f"{Path(surface_name).stem}.obj"

        if not surf_path.exists():
            print(f"Skipping missing file: {surf_path}")
            continue

        coords, faces = mne.read_surface(surf_path)

        mne.write_surface(obj_path, coords, faces, overwrite=overwrite)

        print(f"Converted: {surface_name} -> {obj_path}")

    print("Done.")


if __name__ == "__main__":
    main()