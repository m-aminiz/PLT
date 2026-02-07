import sys, os
from cx_Freeze import setup, Executable

__version__ = "7"

include_files = ["roi.png"]
excludes = ["Tkinter"]
packages = ["tkinter", "numpy", "json", "PIL", "magic", "pydicom", "glob", "queue", "tkscrolledframe", "ttkthemes", "pylibjpeg"]

setup(
    name = "PltApp",
    description='PET Labeling Tool Application',
    version=__version__,
    options = {"build_exe": {
    'packages': packages,
    'include_files': include_files,
    'excludes': excludes,
    'include_msvcr': True,
}},
executables = [Executable("PltApp.py",base="Win32GUI")]
)