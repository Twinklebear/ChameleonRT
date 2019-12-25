// This header is shared across all backends

#ifndef UTIL_TEXTURE_CHANNEL_MASK_H
#define UTIL_TEXTURE_CHANNEL_MASK_H

/* The material's texture_channel_mask stores a packed set of channel
 * indices to track which channel of an RGB texture a scalar value of
 * the material should be fetched from. Each value occupies 2 bits,
 * and stores 0 = R, 1 = G, 2 = B. The ordering of each material
 * parameters channel info in the mask is:
 *
 * - high bit -
 * specular_transmission[20:21]
 * ior[18:19]
 * clearcoat_gloss[16:17]
 * clearcoat[14:15]
 * sheen_tint[12:13]
 * sheen[10:11]
 * anisotropy[8:9]
 * specular_tint[6:7]
 * roughness[4:5]
 * specular[2:3]
 * metallic[0:1]
 * - lo bit -
 */

#define TEXTURE_CHANNEL_BIT_MASK 0x3

#define METALLIC_CHANNEL_BIT 0
#define GET_METALLIC_CHANNEL(mask) (mask & TEXTURE_CHANNEL_BIT_MASK)
#define SET_METALLIC_CHANNEL(mask, x) mask |= x & TEXTURE_CHANNEL_BIT_MASK

#define SPECULAR_CHANNEL_BIT 2
#define GET_SPECULAR_CHANNEL(mask) ((mask >> 2) & TEXTURE_CHANNEL_BIT_MASK)
#define SET_SPECULAR_CHANNEL(mask, x) mask |= (x & TEXTURE_CHANNEL_BIT_MASK) << 2

#define ROUGHNESS_CHANNEL_BIT 4
#define GET_ROUGHNESS_CHANNEL(mask) ((mask >> 4) & TEXTURE_CHANNEL_BIT_MASK)
#define SET_ROUGHNESS_CHANNEL(mask, x) mask |= (x & TEXTURE_CHANNEL_BIT_MASK) << 4

#define SPECULAR_TINT_CHANNEL_BIT 6
#define GET_SPECULAR_TINT_CHANNEL(mask) ((mask >> 6) & TEXTURE_CHANNEL_BIT_MASK)
#define SET_SPECULAR_TINT_CHANNEL(mask, x) mask |= (x & TEXTURE_CHANNEL_BIT_MASK) << 6

#define ANISOTROPY_CHANNEL_BIT 8
#define GET_ANISOTROPY_CHANNEL(mask) ((mask >> 8) & TEXTURE_CHANNEL_BIT_MASK)
#define SET_ANISOTROPY_CHANNEL(mask, x) mask |= (x & TEXTURE_CHANNEL_BIT_MASK) << 8

#define SHEEN_CHANNEL_BIT 10
#define GET_SHEEN_CHANNEL(mask) ((mask >> 10) & TEXTURE_CHANNEL_BIT_MASK)
#define SET_SHEEN_CHANNEL(mask, x) mask |= (x & TEXTURE_CHANNEL_BIT_MASK) << 10

#define SHEEN_TINT_CHANNEL_BIT 12
#define GET_SHEEN_TINT_CHANNEL(mask) ((mask >> 12) & TEXTURE_CHANNEL_BIT_MASK)
#define SET_SHEEN_TINT_CHANNEL(mask, x) mask |= (x & TEXTURE_CHANNEL_BIT_MASK) << 12

#define CLEARCOAT_CHANNEL_BIT 14
#define GET_CLEARCOAT_CHANNEL(mask) ((mask >> 14) & TEXTURE_CHANNEL_BIT_MASK)
#define SET_CLEARCOAT_CHANNEL(mask, x) mask |= (x & TEXTURE_CHANNEL_BIT_MASK) << 14

#define CLEARCOAT_GLOSS_CHANNEL_BIT 16
#define GET_CLEARCOAT_GLOSS_CHANNEL(mask) ((mask >> 16) & TEXTURE_CHANNEL_BIT_MASK)
#define SET_CLEARCOAT_GLOSS_CHANNEL(mask, x) mask |= (x & TEXTURE_CHANNEL_BIT_MASK) << 16

#define IOR_CHANNEL_BIT 18
#define GET_IOR_CHANNEL(mask) ((mask >> 18) & TEXTURE_CHANNEL_BIT_MASK)
#define SET_IOR_CHANNEL(mask, x) mask |= (x & TEXTURE_CHANNEL_BIT_MASK) << 18

#define SPECULAR_TRANSMISSION_CHANNEL_BIT 20
#define GET_SPECULAR_TRANSMISSION_CHANNEL(mask) ((mask >> 20) & TEXTURE_CHANNEL_BIT_MASK)
#define SET_SPECULAR_TRANSMISSION_CHANNEL(mask, x) mask |= (x & TEXTURE_CHANNEL_BIT_MASK) << 20

#endif
