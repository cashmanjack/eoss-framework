# Memory_5: which SM's L1 does the shared traversal unit use?

ncu gives only sum, max and min of L1 counters over the 142 SMs, so the test uses the maxima:
one block (or two) runs the TTU group on chosen SMs (`--smfilter` build, `--sms A,B`), every other
SM's block only loads its inputs and exits. RT core sectors = all L1 sectors minus LSU sectors.
If an SM's node fetches went only to its own L1, the busiest L1 would gain all of them.

    python ../../harness/setup3/run.py BIN file:RAYS zero OUT.json --scene soup:100000 --warps 16 --blocks 142 \
        --sms 0,1 --ncu l1tex__t_sectors,l1tex__t_sectors_pipe_lsu,l1tex__t_sectors_lookup_hit --ncu-out OUT.csv
