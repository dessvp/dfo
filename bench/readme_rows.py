"""README result rows from rank_all.py output.

Usage: python readme_rows.py OUT [DIM:OUT ...]
Ranks exdata/OUT/{P,U}{dim} for every dimension (an optional DIM:OUT
takes that dimension from another run) and prints the archive VBS, development and validation
rows of the README's Table 1.
"""
import re
import subprocess
import sys

out = sys.argv[1]
override = dict(a.split(":") for a in sys.argv[2:])
for d in (2, 3, 5, 10, 20, 40):
    o = override.get(str(d), out)
    r = subprocess.run([sys.executable, "rank_all.py", f"dev=exdata/{o}/P{d}", f"val=exdata/{o}/U{d}", "--top", "1", "--eligible"],
                       capture_output=True, text=True).stdout
    vbs = re.search(r"^\| virtual best \(all archived\) \|(.*)\|$", r, re.M).group(1)
    print(f"| {d}-D | archive VBS |" + vbs + "|")
    for name, label in (("dev", "development"), ("val", "validation")):
        cells = re.search(rf"^\| \*\*{name}\*\* \|(.*)\|$", r, re.M).group(1).split("|")
        fmt = []
        for c in cells:
            v, k = re.match(r"\s*([\d.]+) \(#(\d+)\)", c).groups()
            fmt.append(f"{v} **#1**" if k == "1" else f"{v} #{k}")
        print(f"| {d}-D | {label} | " + " | ".join(fmt) + " |")
