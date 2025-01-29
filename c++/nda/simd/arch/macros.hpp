#pragma once

#define NDA_SHUFFLE_MASK4(x, y, z, t) (((t) << 6) | ((z) << 4) | ((y) << 2) | (x))

#define NDA_SHUFFLE_MASK2(x, y) (((y) << 1) | (x))