#include <metal_common>
#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

kernel void display_image(uint2 tid [[thread_position_in_grid]],
                          texture2d<float, access::read> in [[texture(0)]],
                          texture2d<float, access::write> out [[texture(1)]])
{
    out.write(in.read(tid), tid);
}

