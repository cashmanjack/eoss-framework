"""Generate SM89 TTU encodings with the EoSS encoder.

Rebuilds Cesar's SM120 canonical traversal (data/encodings/ttu_sm120.json)
for SM89, field by field, and writes ttu_sm89.json next to this script.
Run from the repo root:  python ttu/encodings/gen_ttu_sm89.py

Nothing here touches the GPU.
"""
import json, struct, sys
from pathlib import Path

try:
    from eoss.enc.generator import Encoder
except ImportError:  # running from a checkout without the package installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from eoss.enc.generator import Encoder

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SM120 = json.loads((REPO / "data/encodings/ttu_sm120.json").read_text())
CLOSE_BIT_SPEC = 74   # /CLOSE in both the SM89 and SM120 ISA JSON
CLOSE_BIT_CESAR = 72  # what ttu_sm120.json actually sets (unmodeled bit)

enc = Encoder(sm="89")

def build(name, cls, fields, opex):
    ins = enc.build(cls, fields, opex=opex)
    assert ins.valid, (name, ins.failed_conditions)
    lo, hi = struct.unpack("<QQ", ins.bytes)
    return {"name": name, "class": cls, "fields": fields, "opex": opex,
            "lo": f"0x{lo:016x}", "hi": f"0x{hi:016x}"}

# opex values copied from Cesar's SM120 control bits:
# TTUOPEN stall 6 yield, TTUMACROFUSE stall 4, everything else stall 1 yield.
# No scoreboards set (wr=rd=7), exactly as in his sequence.
seq = [build("TTUOPEN", "ttuopen_", {}, 22),
       build("TTUMACROFUSE(8)", "ttumacrofuse_", {"Sb": 8}, 4)]
for off in (0x000, 0x040, 0x050, 0x080):
    seq.append(build(f"TTUST [0x{off:03x}]", "ttust_",
                     {"ImmU16": off, "Rb": "R0", "Rc": "R0"}, 17))
seq.append(build("TTUGO", "ttugo_", {}, 17))
for _ in range(2):
    seq.append(build("TTULD", "ttuld__no_close",
                     {"Pu": "P0", "Rd": "R0", "Rd2": "R0", "ImmU16": 0}, 17))
seq.append(build("TTULD.CLOSE", "ttuld__close",
                 {"Pu": "P0", "Rd": "R0", "Rd2": "R0", "ImmU16": 0}, 17))

# Compare against Cesar's SM120 bytes
ref = SM120["sequences"]["canonical_traversal"]["instructions"]
diffs = []
for mine, his in zip(seq, ref):
    same = (int(mine["lo"], 16), int(mine["hi"], 16)) == (int(his["lo"], 16), int(his["hi"], 16))
    mine["matches_sm120"] = same
    if not same:
        diffs.append({"name": mine["name"], "sm89_spec": [mine["lo"], mine["hi"]],
                      "sm120_cesar": [his["lo"], his["hi"]]})

# Alternative close using Cesar's bit 72, kept for reference only
spec_close = seq[-1]
cesar_close = dict(spec_close, name="TTULD.CLOSE (Cesar bit 72 variant)",
                   hi=f"0x{(int(spec_close['hi'],16) & ~(1 << (CLOSE_BIT_SPEC-64))) | (1 << (CLOSE_BIT_CESAR-64)):016x}")
cesar_close.pop("matches_sm120", None)

out = {
    "schema": "eoss.encodings/v1",
    "sm": "89",
    "generated_by": "ttu/encodings/gen_ttu_sm89.py",
    "comment": ("SM89 rebuild of the SM120 canonical traversal. Opcodes, field "
                "positions and control bits are identical to SM120. TTUST's second "
                "source and TTULD's second destination (bits 71:64) are ordinary "
                "registers on both architectures (named Rc on SM89 and Ra_URc on "
                "SM120, operand type Register on both), even aligned 64 bit pairs "
                "(IDEST_SIZE 64). The close bit is 74 (see close_bit); Cesar's "
                "SM120 file sets bit 72, kept only as an alternative. Corrected "
                "2026-11-10: an earlier comment called SM120's field a uniform "
                "register and the close bit unresolved."),
    "close_bit": {"spec": CLOSE_BIT_SPEC, "cesar_sm120": CLOSE_BIT_CESAR,
                  "status": "resolved: bit 74. Read from SM89 OptiX pipeline code in device memory (Setup_4, driver 580.178.04): the last TTULD of each sequence sets bit 74, bit 72 is never set. Also bit 74 in the sm75 gpuWarmup kernel in libnvoptix."},
    "sequences": {"canonical_traversal": {
        "description": "TTUOPEN -> TTUMACROFUSE(8) -> TTUST x4 -> TTUGO -> TTULD x2 -> TTULD.CLOSE. All operands R0/P0 as placeholders.",
        "instructions": seq}},
    "alternatives": {"ttuld_close_bit72": cesar_close},
    "diff_vs_sm120": diffs,
}
(HERE / "ttu_sm89.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"wrote {HERE/'ttu_sm89.json'}")
for d in diffs:
    print("DIFF", d)
print(f"{sum(m['matches_sm120'] for m in seq)}/{len(seq)} instructions byte identical to SM120")
