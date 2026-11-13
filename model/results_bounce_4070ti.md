# Bounce ray oracle test on the 4070 Ti (io, GPU 0), against predictions written beforehand (model/predictions_bounce_4070ti.md)
Run 2026-11-13. OptiX only (no TTU code), closest hit, bounce rays (origin at the camera hit point, random direction), 2^24 primary rays, 10 launches median, every order equal to pixel order for every ray (0 wrong in 32 runs and in the net runs). Soup scenes of Findings 88 and 96. L2 here is 48 MB (60 SMs), on the RTX 6000 Ada 96 MB (142 SMs). Real scenes (Finding 102) not rerun: their files are not on io.

Speedup over pixel order, trace only (io; triton from Findings 88 and 96 in brackets):

| Scene (GAS / L2 on io) | rays | Morton k4 / k5 | oracle d1 / d2 / d3 / full | oracle full / best Morton | pixel vs shuffled |
|---|---|---|---|---|---|
| 1e5 (6.5 MB, 0.14x) | 10.2 M | 1.09 / 1.10 [1.10 / 1.11] | 1.14 / 1.23 / 1.26 / 1.26 [1.14 / 1.23 / 1.27 / 1.27] | 1.15 [1.14] | 1.57 [1.9] |
| 3e6 (186 MB, 3.9x) | 11.1 M | 1.14 / 1.15 [1.15 / 1.16] | 1.06 / 1.19 / 1.29 / 1.30 [1.04 / 1.16 / 1.25 / 1.27] | 1.13 [1.09] | 1.91 [2.2] |
| 8e6 (0.52 GB, 10.8x) | 11.2 M | 1.15 / 1.16 [1.14 / 1.17] | 1.06 / 1.14 / 1.25 / 1.29 [1.05 / 1.12 / 1.21 / 1.28] | 1.11 [1.09] | 2.17 [2.2] |
| 16e6 (1.04 GB, 21.6x) | 11.3 M | 1.07 / 1.19 [1.12 / 1.16] | 1.05 / 1.13 / 1.13 / 1.31 [1.01 / 1.09 / 1.18 / 1.28] | 1.10 [1.10] | 2.32 [2.5] |

(The 16M k4 value 1.07 is one low launch set: the same sort mode traces at 1.20 in the net run, so read 16M Morton as about 1.19.)

Pixel order trace time (ms): 3.45, 4.87, 5.26, 5.19 (triton about 2.4 to 2.6 for the large scenes: 2.1 to 2.4 times slower with 60 vs 142 SMs).

Net of costs (GPU key generation, CUB sort, gather, trace, scatter; best 4 bit Morton), ms / of pixel order speed: 8.36 / 0.41 (1e5), 9.82 / 0.50 (3e6), 10.30 / 0.51 (8e6), 10.51 / 0.53 (16e6) [triton 0.45 to 0.46]. Against a shuffled launch: 0.64, 0.94, 1.10, 1.16 [triton 1.01 and 1.26 at 8M and 16M]. Sort work (keygen + sort + gather + scatter) is about 4.7 to 5.9 ms on 11 M rays (triton about 3.1): 1.5 to 1.9 times slower.

Verdict on the predictions: (1) oracle 1.2 to 1.35: yes (1.26 to 1.31). (2) Morton 1.08 to 1.20: yes. (3) gap oracle / Morton 1.07 to 1.17, not growing by more than 0.05: yes (1.15, 1.13, 1.11, 1.10; triton 1.14, 1.09, 1.09, 1.10; the largest difference, 0.04, is at 3e6, and the gap falls with scene size here). (4) pixel order 2.0 to 3.0 times faster than shuffled: partly wrong, 1.57 at 1e5 and 1.91 at 3e6 are below the range (the shuffled penalty is smaller on this chip for scenes near the L2 size); 2.17 and 2.32 at 8e6 and 16e6 are in it. (5) pixel order trace 5 to 7 ms: 5.2 to 5.3 ms for the big scenes yes, 3.5 and 4.9 ms for the small ones lower. (6) net 0.45 to 0.6: yes for the three large scenes; 0.41 for 1e5 (sort cost is the same size, the trace is shorter).

Answer: with half the L2 (and the scene 2 times larger relative to it) the oracle gap over Morton does not grow: 1.10 to 1.15 against 1.09 to 1.14. The oracle gain over pixel order stays at 1.26 to 1.31, the same plateau as at 96 MB. One chip, one ray set per scene, one session: differences under about 0.04 are inside the launch to launch noise seen at 16M.
