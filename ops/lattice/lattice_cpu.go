package lattice

import (
	"crypto/sha256"
	"encoding/binary"
)

// CPU fallback implementations

func nttForwardCPU(params NTTParams, coeffs []uint64) ([]uint64, error) {
	n := int(params.N)
	result := make([]uint64, n)
	copy(result, coeffs)

	// Bit-reverse permutation
	bitReverse(result)

	// Cooley-Tukey NTT
	for len := 2; len <= n; len *= 2 {
		wlen := modPow(params.Root, uint64(n/len), params.Modulus)
		for i := 0; i < n; i += len {
			w := uint64(1)
			for j := 0; j < len/2; j++ {
				u := result[i+j]
				v := mulMod(result[i+j+len/2], w, params.Modulus)
				result[i+j] = addMod(u, v, params.Modulus)
				result[i+j+len/2] = subMod(u, v, params.Modulus)
				w = mulMod(w, wlen, params.Modulus)
			}
		}
	}

	return result, nil
}

func nttInverseCPU(params NTTParams, evals []uint64) ([]uint64, error) {
	n := int(params.N)
	result := make([]uint64, n)
	copy(result, evals)

	// Bit-reverse permutation
	bitReverse(result)

	// Gentleman-Sande inverse NTT
	rootInv := modInverse(params.Root, params.Modulus)
	for len := 2; len <= n; len *= 2 {
		wlen := modPow(rootInv, uint64(n/len), params.Modulus)
		for i := 0; i < n; i += len {
			w := uint64(1)
			for j := 0; j < len/2; j++ {
				u := result[i+j]
				v := result[i+j+len/2]
				result[i+j] = addMod(u, v, params.Modulus)
				result[i+j+len/2] = mulMod(subMod(u, v, params.Modulus), w, params.Modulus)
				w = mulMod(w, wlen, params.Modulus)
			}
		}
	}

	// Scale by 1/n
	nInv := modInverse(uint64(n), params.Modulus)
	for i := range result {
		result[i] = mulMod(result[i], nInv, params.Modulus)
	}

	return result, nil
}

func polyMulCPU(params NTTParams, a, b []uint64) ([]uint64, error) {
	// Transform to NTT domain
	aNTT, _ := nttForwardCPU(params, a)
	bNTT, _ := nttForwardCPU(params, b)

	// Pointwise multiplication
	cNTT := make([]uint64, len(aNTT))
	for i := range aNTT {
		cNTT[i] = mulMod(aNTT[i], bNTT[i], params.Modulus)
	}

	// Transform back
	return nttInverseCPU(params, cNTT)
}

func polyAddCPU(modulus uint64, a, b []uint64) ([]uint64, error) {
	result := make([]uint64, len(a))
	for i := range a {
		result[i] = addMod(a[i], b[i], modulus)
	}
	return result, nil
}

func polySubCPU(modulus uint64, a, b []uint64) ([]uint64, error) {
	result := make([]uint64, len(a))
	for i := range a {
		result[i] = subMod(a[i], b[i], modulus)
	}
	return result, nil
}

func batchNTTForwardCPU(params NTTParams, polys [][]uint64) ([][]uint64, error) {
	results := make([][]uint64, len(polys))
	for i, p := range polys {
		r, err := nttForwardCPU(params, p)
		if err != nil {
			return nil, err
		}
		results[i] = r
	}
	return results, nil
}

func batchNTTInverseCPU(params NTTParams, polys [][]uint64) ([][]uint64, error) {
	results := make([][]uint64, len(polys))
	for i, p := range polys {
		r, err := nttInverseCPU(params, p)
		if err != nil {
			return nil, err
		}
		results[i] = r
	}
	return results, nil
}

func sampleNTTCPU(params NTTParams, seed []byte) ([]uint64, error) {
	result := make([]uint64, params.N)
	h := sha256.New()

	for i := uint32(0); i < params.N; i++ {
		h.Reset()
		h.Write(seed)
		binary.Write(h, binary.LittleEndian, i)
		hash := h.Sum(nil)
		result[i] = binary.LittleEndian.Uint64(hash[:8]) % params.Modulus
	}

	return result, nil
}

// Modular arithmetic helpers
func addMod(a, b, m uint64) uint64 {
	sum := a + b
	if sum >= m || sum < a {
		sum -= m
	}
	return sum
}

func subMod(a, b, m uint64) uint64 {
	if a >= b {
		return a - b
	}
	return m - b + a
}

func mulMod(a, b, m uint64) uint64 {
	// Simple but slow - production would use Montgomery reduction
	return (a * b) % m
}

func modPow(base, exp, m uint64) uint64 {
	result := uint64(1)
	base = base % m
	for exp > 0 {
		if exp&1 == 1 {
			result = mulMod(result, base, m)
		}
		exp >>= 1
		base = mulMod(base, base, m)
	}
	return result
}

func modInverse(a, m uint64) uint64 {
	return modPow(a, m-2, m) // Fermat's little theorem
}

func bitReverse(a []uint64) {
	n := len(a)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
}
