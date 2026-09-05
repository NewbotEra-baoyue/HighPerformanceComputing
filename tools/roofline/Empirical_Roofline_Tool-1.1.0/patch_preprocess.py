#!/usr/bin/env python3
import sys

with open("Scripts/preprocess.py", "r") as f:
    content = f.read()

if "precision_value = None" not in content:
    content = content.replace("import sys", "import sys\n\nprecision_value = None")

old = '  if len(m) == 2 and m[0] == "OPENMP_THREADS":\n    threads = int(m[1])\n  if len(m) == 2 and m[0] == "MPI_PROCS":\n    procs = int(m[1])'
new = '  if len(m) == 2 and m[0] == "OPENMP_THREADS":\n    threads = int(m[1])\n  if len(m) == 2 and m[0] == "MPI_PROCS":\n    procs = int(m[1])\n  if len(m) == 2 and m[0] == "FLOPS":\n    global precision_value\n    precision_value = int(m[1])'

if old in content:
    content = content.replace(old, new)
else:
    print("WARNING: block not found"); sys.exit(1)

if "int(precision.value)" in content:
    content = content.replace("int(precision.value)", "int(precision_value) if precision_value is not None else 1")

with open("Scripts/preprocess.py", "w") as f:
    f.write(content)
print("Patched successfully!")