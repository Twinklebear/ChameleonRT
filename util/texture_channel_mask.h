// This header is shared across all backends

#ifndef UTIL_TEXTURE_CHANNEL_MASK_H
#define UTIL_TEXTURE_CHANNEL_MASK_H

/* The material's float parameters (and the r component of the color)
 * can be a positive scalar, representing the input value, or if
 * negative correspond to a texture input. The bits are the interpreted as:
 *
 * [31]: 1, sign bit indicating this is a handle
 * [29:30]: 2 bits indicating the texture channel to use (base_color is
 *          assumed to use all channels and this is ignored)
 * [0:28]: texture ID
 */

#define TEXTURED_PARAM_MASK 0x80000000
#define IS_TEXTURED_PARAM(x) ((x) & 0x80000000)

#define GET_TEXTURE_CHANNEL(x) (((x) >> 29) & 0x3)
#define SET_TEXTURE_CHANNEL(x, c) x |= (c & 0x3) << 29

#define GET_TEXTURE_ID(x) ((x) & 0x1fffffff)
#define SET_TEXTURE_ID(x, i) (x |= i & 0x1fffffff)

#endif
