# rtobj

An example of OBJ rendering with OSPRay, Embree and OptiX.
Uses [tinyobjloader]() to load OBJ files.

## Ray Tracing Backends  

The currently implemented backends are: OSPRay, Embree and OptiX.
When running the program, you can pick which backend you want from
those you compiled with by specifying it as the first argument on
the command line:

```
./rtobj <backend> <mesh.obj>
```

### OSPRay

Dependencies: [OSPRay](http://www.ospray.org/).

To build the OSPRay backend run CMake with:

```
cmake .. -DENABLE_OSPRAY=ON -Dospray_DIR=<path to osprayConfig.cmake>
```

You can then pass `-ospray` to use the OSPRay backend.

### Embree

Dependencies: [Embree](https://embree.github.io/), [TBB](https://www.threadingbuildingblocks.org/).

To build the Embree backend run CMake with:

```
cmake .. -DENABLE_EMBREE=ON -Dembree_DIR=<path to embree-config.cmake> \
    -DTBB_DIR=<path to root of TBB install>
```

You can then pass `-embree` to use the Embree backend.

### OptiX

Dependencies: [OptiX 6](https://developer.nvidia.com/optix), [CUDA 10](https://developer.nvidia.com/cuda-zone).

To build the OptiX backend run CMake with:

```
cmake .. -DENABLE_OPTIX=ON
```

You can then pass `-optix` to use the OptiX backend.

If CMake doesn't find your install of OptiX you can tell it where
it's installed with `-DOptiX_INSTALL_DIR`.

