// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lattice

import (
	"github.com/luxfi/accel"
)

// MLDSANTTBatch performs the in-place forward (inverse=false) or inverse
// (inverse=true) Number-Theoretic Transform over Z_q[X]/(X^256 + 1) with
// q = 8380417 (the FIPS 204 ML-DSA prime) on every polynomial in polys.
//
// Each polys[i] MUST have length 256. The transform writes back into the
// input slice in-place.
//
// Byte-equal to PQCLEAN_MLDSA65_CLEAN_ntt (forward) and
// PQCLEAN_MLDSA65_CLEAN_invntt_tomont (inverse). The dispatcher engages
// the GPU substrate when len(polys) >= accel.MLDSABatchThreshold AND the
// substrate ships an implementation; otherwise this falls through to the
// per-poly Go oracle below, which is the canonical CPU reference for
// Pulsar's byte-equality regression test.
func MLDSANTTBatch(polys [][]int32, inverse bool) error {
	n := len(polys)
	if n == 0 {
		return ErrInvalidInput
	}
	for _, p := range polys {
		if len(p) != mldsaN {
			return ErrInvalidDegree
		}
	}

	// Engage the substrate when we have enough work and the runtime
	// actually has a backend wired. The accel-level entry handles its
	// own threshold + Available() gate; we just call it.
	if err := accel.LatticeNTTMLDSABatch(polys, inverse); err == nil {
		return nil
	}

	// CPU oracle path. Byte-equal to PQCLEAN_MLDSA65_CLEAN_ntt /
	// invntt_tomont by construction.
	if inverse {
		for _, p := range polys {
			mldsaINTTInPlace(p)
		}
	} else {
		for _, p := range polys {
			mldsaNTTInPlace(p)
		}
	}
	return nil
}

// MLDSANTTForward is the single-poly convenience wrapper around
// MLDSANTTBatch. Returns ErrInvalidDegree if len(coeffs) != 256.
func MLDSANTTForward(coeffs []int32) error {
	if len(coeffs) != mldsaN {
		return ErrInvalidDegree
	}
	mldsaNTTInPlace(coeffs)
	return nil
}

// MLDSANTTInverse is the single-poly convenience wrapper around
// MLDSANTTBatch (inverse direction).
func MLDSANTTInverse(coeffs []int32) error {
	if len(coeffs) != mldsaN {
		return ErrInvalidDegree
	}
	mldsaINTTInPlace(coeffs)
	return nil
}

// =============================================================================
// PQClean ML-DSA-65 NTT — Go oracle, byte-equal to clean/ntt.c.
//
// Constants:
//   q     = 8380417
//   qinv  = 58728449   (q^-1 mod 2^32)
//   N     = 256
//
// zetas[256] is the PQCLEAN_MLDSA65_CLEAN_zetas table (FIPS 204 reference).
// Identical bytes shipped by lux-private/gpu-kernels/ops/lattice/ntt_mldsa.
// =============================================================================

const (
	mldsaQ    int32  = 8380417
	mldsaQInv uint32 = 58728449
	mldsaN           = 256

	// f = (R^2 mod q) * (q-1)^{-1} mod q for invNTT_tomont multiplication.
	// Direct from PQCLEAN_MLDSA65_CLEAN_invntt_tomont.
	mldsaF int32 = 41978
)

// mldsaZetas is the PQCLEAN_MLDSA65_CLEAN_zetas array — bytes identical to
// the kMldsaZetas table embedded in every backend kernel
// (lux-private/gpu-kernels/ops/lattice/ntt_mldsa/{cuda.cu,metal.metal,
// hip.cu,wgsl.wgsl}). Keeping ONE Go-level copy here is the source of
// truth for the CPU oracle; the per-backend embed copies are mechanical
// duplicates required by GPU compile units.
var mldsaZetas = [mldsaN]int32{
	0, 25847, -2608894, -518909, 237124, -777960, -876248, 466468,
	1826347, 2353451, -359251, -2091905, 3119733, -2884855, 3111497, 2680103,
	2725464, 1024112, -1079900, 3585928, -549488, -1119584, 2619752, -2108549,
	-2118186, -3859737, -1399561, -3277672, 1757237, -19422, 4010497, 280005,
	2706023, 95776, 3077325, 3530437, -1661693, -3592148, -2537516, 3915439,
	-3861115, -3043716, 3574422, -2867647, 3539968, -300467, 2348700, -539299,
	-1699267, -1643818, 3505694, -3821735, 3507263, -2140649, -1600420, 3699596,
	811944, 531354, 954230, 3881043, 3900724, -2556880, 2071892, -2797779,
	-3930395, -1528703, -3677745, -3041255, -1452451, 3475950, 2176455, -1585221,
	-1257611, 1939314, -4083598, -1000202, -3190144, -3157330, -3632928, 126922,
	3412210, -983419, 2147896, 2715295, -2967645, -3693493, -411027, -2477047,
	-671102, -1228525, -22981, -1308169, -381987, 1349076, 1852771, -1430430,
	-3343383, 264944, 508951, 3097992, 44288, -1100098, 904516, 3958618,
	-3724342, -8578, 1653064, -3249728, 2389356, -210977, 759969, -1316856,
	189548, -3553272, 3159746, -1851402, -2409325, -177440, 1315589, 1341330,
	1285669, -1584928, -812732, -1439742, -3019102, -3881060, -3628969, 3839961,
	2091667, 3407706, 2316500, 3817976, -3342478, 2244091, -2446433, -3562462,
	266997, 2434439, -1235728, 3513181, -3520352, -3759364, -1197226, -3193378,
	900702, 1859098, 909542, 819034, 495491, -1613174, -43260, -522500,
	-655327, -3122442, 2031748, 3207046, -3556995, -525098, -768622, -3595838,
	342297, 286988, -2437823, 4108315, 3437287, -3342277, 1735879, 203044,
	2842341, 2691481, -2590150, 1265009, 4055324, 1247620, 2486353, 1595974,
	-3767016, 1250494, 2635921, -3548272, -2994039, 1869119, 1903435, -1050970,
	-1333058, 1237275, -3318210, -1430225, -451100, 1312455, 3306115, -1962642,
	-1279661, 1917081, -2546312, -1374803, 1500165, 777191, 2235880, 3406031,
	-542412, -2831860, -1671176, -1846953, -2584293, -3724270, 594136, -3776993,
	-2013608, 2432395, 2454455, -164721, 1957272, 3369112, 185531, -1207385,
	-3183426, 162844, 1616392, 3014001, 810149, 1652634, -3694233, -1799107,
	-3038916, 3523897, 3866901, 269760, 2213111, -975884, 1717735, 472078,
	-426683, 1723600, -1803090, 1910376, -1667432, -1104333, -260646, -3833893,
	-2939036, -2235985, -420899, -2286327, 183443, -976891, 1612842, -3545687,
	-554416, 3919660, -48306, -1362209, 3937738, 1400424, -846154, 1976782,
}

// mldsaMontReduce — PQCLEAN_MLDSA65_CLEAN_montgomery_reduce.
//
//	t := int32((uint64(a) & 0xFFFFFFFF) * uint64(mldsaQInv))
//	return int32((a - int64(t) * int64(mldsaQ)) >> 32)
func mldsaMontReduce(a int64) int32 {
	t := int32(uint32(a) * mldsaQInv)
	return int32((a - int64(t)*int64(mldsaQ)) >> 32)
}

// mldsaNTTInPlace — direct translation of PQCLEAN_MLDSA65_CLEAN_ntt.
// Cooley-Tukey forward NTT in Montgomery domain, 8 levels.
func mldsaNTTInPlace(a []int32) {
	k := uint(0)
	for length := uint(128); length > 0; length >>= 1 {
		for start := uint(0); start < mldsaN; start = start + 2*length {
			k++
			zeta := int64(mldsaZetas[k])
			for j := start; j < start+length; j++ {
				t := mldsaMontReduce(zeta * int64(a[j+length]))
				a[j+length] = a[j] - t
				a[j] = a[j] + t
			}
		}
	}
}

// mldsaINTTInPlace — direct translation of
// PQCLEAN_MLDSA65_CLEAN_invntt_tomont. Gentleman-Sande inverse NTT in
// Montgomery domain followed by multiplication by f = R^2 * (q-1)^-1.
func mldsaINTTInPlace(a []int32) {
	k := uint(256)
	for length := uint(1); length < mldsaN; length <<= 1 {
		for start := uint(0); start < mldsaN; start = start + 2*length {
			k--
			zeta := -int64(mldsaZetas[k])
			for j := start; j < start+length; j++ {
				t := a[j]
				a[j] = t + a[j+length]
				a[j+length] = t - a[j+length]
				a[j+length] = mldsaMontReduce(zeta * int64(a[j+length]))
			}
		}
	}
	for i := 0; i < mldsaN; i++ {
		a[i] = mldsaMontReduce(int64(mldsaF) * int64(a[i]))
	}
}
