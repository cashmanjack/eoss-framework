# Flag, mask, tmax encoding in the OptiX TTUST words: predictions (2026-11-16, tethys, before any run)

Method: memdump (our own OptiX launch, no TTU code of ours) with one raygen variant per process, compile time
constants, read 64 KB below our kernel, diff against the base variant (flags NONE, mask 255, tmax 1e16).
Reading: code differences between variants = where a constant lives in raygen; the traversal function bytes
should be IDENTICAL across variants (flags are run time values in R37).

Predictions:
1. Traversal function (about fnaddr-0x10B00 to the group) byte identical in all variants.
2. Raygen differs only in constants: R37 = (flags << 16) | (mask << 8) | 1 for every flag (DISABLE_ANYHIT 1, ENFORCE_ANYHIT 2,
   TERMINATE 4, DISABLE_CLOSESTHIT 8, CULL_BACK 16, CULL_FRONT 32, CULL_OPAQUE 64, CULL_NON_OPAQUE 128), checked per flag.
3. tmax: only the float constant of R17 changes (1e16 = 0x5A0E1BCA; 0.05 = 0x3D4CCCCD; 2.0 = 0x40000000). No other byte.
4. store_words.py (computed words): flags change only bits of word 0 of stores 0 and 0x80 (cull 0x30 bits and terminate 0x80, as
   already computed); mask changes the top byte of word 0 and bits 23:16 of word 2; tmax does not appear in stores 0 and 0x80.
   tmax sits in the high half of stores 0x50 and 0xD0 (R17), as in protocol.md.
5. Not predicted to differ: DISABLE_ANYHIT, ENFORCE_ANYHIT, CULL_OPAQUE, CULL_NON_OPAQUE words (store_words.py says they change
   only R69 bits through c[3][0x34] paths, not through flags 1, 2, 64, 128): if the computed words do not change for them, the
   flag acts only through R37 and a later stage, not through these stores.
