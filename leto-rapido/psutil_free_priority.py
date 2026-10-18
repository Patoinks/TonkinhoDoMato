"""Imported by worker processes: below-normal priority (the owner's rule), no psutil needed."""
import sys

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
