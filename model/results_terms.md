## Short query (TPC unit capacity) term
RTX 6000 Ada, tri1, one SM 0.1004 rays/cycle, U set from the 4070 Ti ratio (1.377 x one SM): G 32 pred 3.214 meas 3.224 (-0%), G 71 pred 7.131 meas 7.118 (+0%), G 100 pred 8.228 meas 8.083 (+2%), G 120 pred 8.985 meas 8.713 (+3%), G 142 pred 9.817 meas 10.882 (-10%)
  triton alone: U = 0.1533 per TPC = 1.526 x one SM (4070 Ti: 1.377 x): unit capacity ratio differs by 11% between the two chips
RTX 6000 Ada, par64, one SM 0.0178 rays/cycle, U set from the 4070 Ti ratio (1.04 x one SM): G 32 pred 0.569 meas 0.571 (-0%), G 71 pred 1.262 meas 1.256 (+1%), G 100 pred 1.283 meas 1.263 (+2%), G 120 pred 1.297 meas 1.233 (+5%), G 142 pred 1.313 meas 1.307 (+0%)
4070 Ti, tri1, U set from triton (1.529 x one SM): G 30 pred 30.0 meas 30.0, G 60 pred 45.9 meas 41.3 (+11%)
4070 Ti, par64, U set from triton (1.529 x one SM): G 30 pred 30.0 meas 29.8, G 60 pred 31.2 meas 31.1 (+0%)

## Payload / attribute term (OptiX kernel on the 4070 Ti, Finding 112)
registers per thread, formula minus ncu over all 25 configurations: mean -0.5, max abs 4.8
triangles rand: payload 0 -> 32 took 0.729 -> 0.753 ms, warps 13.6 -> 10.2, exponent k = 0.11 (set from this pair)
triangles coh: payload 0 -> 32 took 0.221 -> 0.271 ms, warps 13.6 -> 10.2, exponent k = 0.70 (set from this pair)
test on the other triangle configurations (k fixed): rand p16 a2 pred +0.0% meas +0.0%, rand p16 a8 pred +3.4% meas +3.3%, rand p32 a8 pred +3.4% meas +3.0%, coh p16 a2 pred +0.0% meas +0.2%, coh p16 a8 pred +22.4% meas +22.4%, coh p32 a8 pred +22.4% meas +23.2%
custom primitives rand (k from the triangle run of the same ray set 0.11): payload 2 -> 32 predicted +3%, measured +22%
custom primitives coh (k from the triangle run of the same ray set 0.70): payload 2 -> 32 predicted +22%, measured +26%
Reading: the register and occupancy part (registers -> resident warps) is predicted; the slowdown needs the regime exponent k (0.1 random triangles, 0.7 coherent), set from one pair, and the custom primitive random case (intersection program on the SM, issue bound) is NOT predicted by the triangle k.
