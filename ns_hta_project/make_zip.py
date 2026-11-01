import zipfile
import os

zip_path = r"C:\Users\DELL\.gemini\antigravity\scratch\ns_hta_project.zip"
src_dir = r"C:\Users\DELL\.gemini\antigravity\scratch\ns_hta_project"

# Remove existing zip if any
if os.path.exists(zip_path):
    try:
        os.remove(zip_path)
    except Exception:
        pass

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(src_dir):
        # Ignore virtual env and pycache
        if any(ignored in root for ignored in [".venv", "venv", "__pycache__", ".git"]):
            continue
        for file in files:
            if file in ["make_zip.py", "ns_hta_project.zip"]:
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, src_dir)
            z.write(full_path, rel_path)

print(f"Clean zip created! Size: {os.path.getsize(zip_path) / 1024:.1f} KB, Total files: {len(zipfile.ZipFile(zip_path).namelist())}")
