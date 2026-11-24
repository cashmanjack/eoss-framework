# What the stores at 0xC0, 0xD0, 0x40 and 0x50 carry (predictions model/predictions_ray_stores.md, committed first, dd9c45f)

2026-11-24, triton GPU 1, one warp, one variant at a time, no other GPU compute process before each run (nvidia-smi PIDs and `who`), run.py gate under timeout -k 2 30. 16 variants, each the checked single session build with one MOV32I replacing the last NOP before the TTUOPEN (check_seq accepts them by hash: base passes, exactly that one instruction differs, it decodes with 0 failed checks). All 16 ran: no fault, no timeout, no hang; all lanes agreed in every run. Hit ray scene tri1 (base: hit, t 1.0, u 0.25, v 0.5, 1,586 cycles) and miss ray scene par:1024 (base: miss, about 100,500 cycles); one launch for the result registers, 30 launches for the cycles (medians). Constants: words 0 and 1 set to 0.1, word 2 of an origin store -2.0, of a direction store 2.0, word 3 of 0xC0 and 0x40 (tmin) 1.5, of 0xD0 and 0x50 (tmax) 0.5.

| store | word (register) | hit ray result | miss ray scene | named |
|---|---|---|---|---|
| 0xC0 | 0 (R10) | u 0.25 to 0.35 | unchanged | origin x |
| 0xC0 | 1 (R11) | u 0.20, v 0.60 | unchanged | origin y |
| 0xC0 | 2 (R8) | t 1.0 to 2.0 | unchanged | origin z |
| 0xC0 | 3 (R9) | unchanged | unchanged | not read as tmin (at 1.5) |
| 0xD0 | 0 (R14) | u 0.35 | unchanged | direction x |
| 0xD0 | 1 (R15) | u 0.20, v 0.60 | became a HIT (t 0.01), session 2,295 cycles instead of 100,599 | direction y |
| 0xD0 | 2 (R12) | t 1.0 to 0.5 | unchanged | direction z |
| 0xD0 | 3 (R13) | unchanged | unchanged | not read as tmax (at 0.5) |
| 0x40 | 0, 1, 2 (R34, R35, R32) | unchanged | unchanged | origin words not used at the changes tried |
| 0x40 | 3 (R33) | hit lost, 1,585 to 1,209 cycles | 100,595 to 26,693 cycles | tmin |
| 0x50 | 0, 1, 2 (R38, R39, R36) | unchanged | unchanged | direction words not used at the changes tried |
| 0x50 | 3 (R37) | hit lost, 1,594 to 1,226 cycles | 100,595 to 23,380 cycles | tmax |

Prediction scoring (H1: 0xC0 and 0xD0 the ray for the triangle test, 0x40 and 0x50 only the instance level): held for words 0 to 2 of 0xC0 and 0xD0 (u, v, t as predicted, 6 of 6), held for words 0 to 2 of 0x40 and 0x50 (no change, 6 of 6), held for 0x40 and 0x50 word 3 (hit lost, 2 of 2); MISSED for 0xC0 word 3 and 0xD0 word 3 (I predicted the hit would be lost; nothing changed). The session length prediction (within 3 percent) held except where the hit is lost (shorter) and the direction y case on the miss scene (it hit). No fault, no hang (predicted).
Limits: "not used" means no change at the single value tried; a large change of an unused word (for instance an origin word of 0x40 at 100.0) was not tried, so an instance level box test that only fails for far away rays is not excluded. A word 3 of 0xC0 and 0xD0 that is read only when below the real tmin or above the real tmax was not tested either. The register pairs are those of our remapped build (R8 to R15, R32 to R39); what OptiX's own code has in those words is the protocol.md table (origin and direction in R56, R57, R52, R53 and R46, R47, R30, R31 for 0xC0 and 0xD0; the same ray values, tmin R13 and tmax R17 for 0x40 and 0x50), unchanged here and consistent: in OptiX's code the 0xC0 and 0xD0 stores take their words from different registers than 0x40 and 0x50 but the same values, so which pair the unit really reads is only seen by changing one of them, as here.
