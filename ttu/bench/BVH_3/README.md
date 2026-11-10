# BVH_3: custom primitives whose intersection always returns false

Plain OptiX (no patched code). Same geometry as setup3's `par:N:s|o` scenes, as triangles or as
custom primitives (each triangle's box); the intersection program never reports a hit.

    nvcc -O2 -arch=sm_89 -ptx -I$HOME/optix_sdk/include -o BUILD/bvh3_dev.ptx bvh3_dev.cu
    nvcc -O2 -arch=sm_89 -I$HOME/optix_sdk/include -o BUILD/bvh3 bvh3.cu -lcuda
    python3 run.py BUILD WORK
    /usr/bin/python3 run.py --report WORK
