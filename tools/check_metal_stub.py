"""Check that builds without Metal export every Metal symbol that Warp binds.

``warp/_src/context.py`` binds the Metal runtime's entry points with ctypes whenever the library loads, so a
library built without Metal (CMake on Linux or Windows, where ``metal_stub.cpp`` replaces ``metal.mm``) must
still define all of them, or importing Warp fails there. This compares three lists:

* the names with ``metal`` in them that ``warp/_src`` looks up on the native library (``core.<name>``),
* the functions ``warp/native/metal.h`` declares with ``WP_API``,
* the functions ``warp/native/metal_stub.cpp`` defines,

and exits with status 1, naming the missing symbols, if a bound or declared symbol has no stub. Usage:

    python tools/check_metal_stub.py [REPO_ROOT]
"""

import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent)
python_sources = sorted((root / "warp" / "_src").rglob("*.py"))
header = (root / "warp" / "native" / "metal.h").read_text()
stub = (root / "warp" / "native" / "metal_stub.cpp").read_text()

bound = set()
for path in python_sources:
    bound |= {m for m in re.findall(r"\bcore\.(\w+)", path.read_text()) if "metal" in m}

declared = set(re.findall(r"WP_API\b[^;(]*?\b(\w+)\s*\(", header))

# a function definition: a name, its parameter list, then a body (not a declaration ending in ';')
defined = set(re.findall(r"\b(\w+)\s*\([^;{}]*\)\s*(?:const\s*)?\{", stub))

missing_bound = sorted(bound - defined)
missing_declared = sorted(declared - defined)
print(
    f"{len(bound)} Metal symbols bound by warp/_src, {len(declared)} declared in metal.h, {len(defined)} defined in metal_stub.cpp"
)
if missing_bound or missing_declared:
    for name in missing_bound:
        print(f"missing from metal_stub.cpp: {name} (bound in warp/_src)")
    for name in sorted(set(missing_declared) - set(missing_bound)):
        print(f"missing from metal_stub.cpp: {name} (declared in metal.h)")
    sys.exit(1)
print("metal_stub.cpp defines every bound and declared Metal symbol")
