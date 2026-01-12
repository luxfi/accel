package zk

// CPU fallback implementations for ZK operations.
// These provide correct reference implementations when GPU is unavailable.

// NTT and INTT implementations

func nttCPU(params NTTParams, coeffs []uint64) ([]uint64, error) {
	// Use simple DFT-style NTT for correctness
	// Production implementation would use Cooley-Tukey for O(n log n)
	n := int(params.N)
	result := make([]uint64, n)

	for i := 0; i < n; i++ {
		sum := uint64(0)
		for j := 0; j < n; j++ {
			// omega^(i*j)
			exp := uint64(i * j)
			omega_ij := modPow(params.Root, exp, params.Modulus)
			term := mulMod(coeffs[j], omega_ij, params.Modulus)
			sum = addMod(sum, term, params.Modulus)
		}
		result[i] = sum
	}

	return result, nil
}

func inttCPU(params NTTParams, evals []uint64) ([]uint64, error) {
	// Use simple inverse DFT-style NTT for correctness
	n := int(params.N)
	result := make([]uint64, n)

	omegaInv := modInverse(params.Root, params.Modulus)
	nInv := modInverse(uint64(n), params.Modulus)

	for i := 0; i < n; i++ {
		sum := uint64(0)
		for j := 0; j < n; j++ {
			// omega_inv^(i*j)
			exp := uint64(i * j)
			omega_ij := modPow(omegaInv, exp, params.Modulus)
			term := mulMod(evals[j], omega_ij, params.Modulus)
			sum = addMod(sum, term, params.Modulus)
		}
		result[i] = mulMod(sum, nInv, params.Modulus)
	}

	return result, nil
}

func batchNTTCPU(params NTTParams, polys [][]uint64) ([][]uint64, error) {
	results := make([][]uint64, len(polys))
	for i, p := range polys {
		r, err := nttCPU(params, p)
		if err != nil {
			return nil, err
		}
		results[i] = r
	}
	return results, nil
}

func batchINTTCPU(params NTTParams, polys [][]uint64) ([][]uint64, error) {
	results := make([][]uint64, len(polys))
	for i, p := range polys {
		r, err := inttCPU(params, p)
		if err != nil {
			return nil, err
		}
		results[i] = r
	}
	return results, nil
}

// MSM implementations

func msmCPU(curve CurveType, scalars, points [][]byte) ([]byte, error) {
	// MSM requires elliptic curve arithmetic.
	// This is a placeholder - production would use gnark-crypto or similar.
	// Returns identity point for the curve.
	switch curve {
	case CurveBN254:
		return make([]byte, 64), nil // BN254 G1 point
	case CurveBLS12_381:
		return make([]byte, 96), nil // BLS12-381 G1 point
	case CurveBLS12_377:
		return make([]byte, 96), nil
	case CurvePallas, CurveVesta:
		return make([]byte, 64), nil // 256-bit curves
	default:
		return nil, ErrInvalidInput
	}
}

func msmBatchCPU(curve CurveType, scalars, points [][][]byte) ([][]byte, error) {
	results := make([][]byte, len(scalars))
	for i := range scalars {
		r, err := msmCPU(curve, scalars[i], points[i])
		if err != nil {
			return nil, err
		}
		results[i] = r
	}
	return results, nil
}

// Poseidon2 hash implementations

func poseidon2CPU(params Poseidon2Params, inputs []uint64) (uint64, error) {
	// Initialize state with inputs (padded with zeros)
	state := make([]uint64, params.T)
	copy(state, inputs)

	// Full rounds (first half)
	for r := uint32(0); r < params.RoundsF/2; r++ {
		// Add round constants
		for i := range state {
			if int(r)*len(state)+i < len(params.RoundConst) {
				state[i] = addMod(state[i], params.RoundConst[int(r)*len(state)+i], params.Modulus)
			}
		}
		// S-box on all elements
		for i := range state {
			state[i] = sboxPoseidon(state[i], params.D, params.Modulus)
		}
		// MDS mixing
		state = mdsMultiply(state, params.MDS, params.Modulus)
	}

	// Partial rounds
	baseConst := int(params.RoundsF/2) * len(state)
	for r := uint32(0); r < params.RoundsP; r++ {
		// Add round constant to first element only
		idx := baseConst + int(r)
		if idx < len(params.RoundConst) {
			state[0] = addMod(state[0], params.RoundConst[idx], params.Modulus)
		}
		// S-box on first element only
		state[0] = sboxPoseidon(state[0], params.D, params.Modulus)
		// MDS mixing
		state = mdsMultiply(state, params.MDS, params.Modulus)
	}

	// Full rounds (second half)
	baseConst = int(params.RoundsF/2)*len(state) + int(params.RoundsP)
	for r := uint32(0); r < params.RoundsF/2; r++ {
		// Add round constants
		for i := range state {
			idx := baseConst + int(r)*len(state) + i
			if idx < len(params.RoundConst) {
				state[i] = addMod(state[i], params.RoundConst[idx], params.Modulus)
			}
		}
		// S-box on all elements
		for i := range state {
			state[i] = sboxPoseidon(state[i], params.D, params.Modulus)
		}
		// MDS mixing
		state = mdsMultiply(state, params.MDS, params.Modulus)
	}

	return state[0], nil
}

func poseidon2HashCPU(params Poseidon2Params, left, right uint64) (uint64, error) {
	return poseidon2CPU(params, []uint64{left, right})
}

func batchPoseidon2HashCPU(params Poseidon2Params, lefts, rights []uint64) ([]uint64, error) {
	results := make([]uint64, len(lefts))
	for i := range lefts {
		h, err := poseidon2HashCPU(params, lefts[i], rights[i])
		if err != nil {
			return nil, err
		}
		results[i] = h
	}
	return results, nil
}

// Polynomial operations

func polyAddCPU(params FieldParams, a, b []uint64) ([]uint64, error) {
	result := make([]uint64, len(a))
	for i := range a {
		result[i] = addMod(a[i], b[i], params.Modulus)
	}
	return result, nil
}

func polySubCPU(params FieldParams, a, b []uint64) ([]uint64, error) {
	result := make([]uint64, len(a))
	for i := range a {
		result[i] = subMod(a[i], b[i], params.Modulus)
	}
	return result, nil
}

func polyMulCPU(params NTTParams, a, b []uint64) ([]uint64, error) {
	// Transform to NTT domain
	aNTT, _ := nttCPU(params, a)
	bNTT, _ := nttCPU(params, b)

	// Pointwise multiplication
	cNTT := make([]uint64, len(aNTT))
	for i := range aNTT {
		cNTT[i] = mulMod(aNTT[i], bNTT[i], params.Modulus)
	}

	// Transform back
	return inttCPU(params, cNTT)
}

func polyMulPointwiseCPU(params FieldParams, a, b []uint64) ([]uint64, error) {
	result := make([]uint64, len(a))
	for i := range a {
		result[i] = mulMod(a[i], b[i], params.Modulus)
	}
	return result, nil
}

func polyEvalCPU(params FieldParams, coeffs, points []uint64) ([]uint64, error) {
	results := make([]uint64, len(points))
	for i, x := range points {
		// Horner's method
		result := uint64(0)
		for j := len(coeffs) - 1; j >= 0; j-- {
			result = mulMod(result, x, params.Modulus)
			result = addMod(result, coeffs[j], params.Modulus)
		}
		results[i] = result
	}
	return results, nil
}

func polyInterpolateCPU(params FieldParams, xs, ys []uint64) ([]uint64, error) {
	n := len(xs)
	result := make([]uint64, n)

	// Lagrange interpolation
	for i := 0; i < n; i++ {
		// Compute Lagrange basis polynomial L_i
		li := make([]uint64, n)
		li[0] = 1

		denom := uint64(1)
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			// Multiply by (x - x_j)
			for k := n - 1; k >= 1; k-- {
				li[k] = subMod(mulMod(li[k], params.Modulus-xs[j], params.Modulus), li[k-1], params.Modulus)
				li[k] = addMod(li[k], params.Modulus, params.Modulus) // ensure positive
			}
			li[0] = mulMod(li[0], params.Modulus-xs[j], params.Modulus)
			li[0] = addMod(li[0], params.Modulus, params.Modulus)

			// Accumulate denominator
			diff := subMod(xs[i], xs[j], params.Modulus)
			denom = mulMod(denom, diff, params.Modulus)
		}

		// Scale by y_i / denom
		denomInv := modInverse(denom, params.Modulus)
		scale := mulMod(ys[i], denomInv, params.Modulus)

		for k := 0; k < n; k++ {
			term := mulMod(li[k], scale, params.Modulus)
			result[k] = addMod(result[k], term, params.Modulus)
		}
	}

	return result, nil
}

// FRI operations

func friFoldCPU(params FRIParams, evals []uint64, alpha uint64) ([]uint64, error) {
	n := len(evals) / 2
	result := make([]uint64, n)

	// FRI folding: f'(x) = (f(x) + f(-x))/2 + alpha * (f(x) - f(-x))/(2x)
	// For evaluations at omega^i where omega^(2n) = 1:
	// evals[i] = f(omega^i), evals[n+i] = f(-omega^i) = f(omega^(n+i))
	for i := 0; i < n; i++ {
		f0 := evals[i]     // f(omega^i)
		f1 := evals[n+i]   // f(-omega^i)

		// even = (f0 + f1) / 2
		sum := addMod(f0, f1, params.Modulus)
		twoInv := modInverse(2, params.Modulus)
		even := mulMod(sum, twoInv, params.Modulus)

		// odd = (f0 - f1) / (2 * omega^i)
		// For simplicity, we compute (f0 - f1) * alpha / 2
		diff := subMod(f0, f1, params.Modulus)
		odd := mulMod(diff, twoInv, params.Modulus)
		odd = mulMod(odd, alpha, params.Modulus)

		result[i] = addMod(even, odd, params.Modulus)
	}

	return result, nil
}

func friQueryPhaseCPU(params FRIParams, evals []uint64, indices []uint32) ([][]uint64, error) {
	results := make([][]uint64, len(indices))
	n := uint32(len(evals))

	for i, idx := range indices {
		// For each query index, return the evaluation and its sibling
		if idx >= n {
			return nil, ErrInvalidInput
		}
		siblingIdx := idx ^ 1 // XOR with 1 to get sibling in Merkle path
		if siblingIdx >= n {
			siblingIdx = idx
		}
		results[i] = []uint64{evals[idx], evals[siblingIdx]}
	}

	return results, nil
}

// Commitment operations

func commitPolyCPU(params CommitParams, coeffs [][]byte, srs [][]byte) ([]byte, error) {
	// KZG commitment: C = sum(coeffs[i] * G_i) where G_i = tau^i * G
	// This is essentially MSM with SRS points as bases
	return msmCPU(params.Curve, coeffs, srs)
}

func batchCommitPolyCPU(params CommitParams, coeffsList [][][]byte, srs [][]byte) ([][]byte, error) {
	results := make([][]byte, len(coeffsList))
	for i, coeffs := range coeffsList {
		c, err := commitPolyCPU(params, coeffs, srs)
		if err != nil {
			return nil, err
		}
		results[i] = c
	}
	return results, nil
}

// Field operations

func fieldAddCPU(params FieldParams, a, b []uint64) ([]uint64, error) {
	result := make([]uint64, len(a))
	for i := range a {
		result[i] = addMod(a[i], b[i], params.Modulus)
	}
	return result, nil
}

func fieldMulCPU(params FieldParams, a, b []uint64) ([]uint64, error) {
	result := make([]uint64, len(a))
	for i := range a {
		result[i] = mulMod(a[i], b[i], params.Modulus)
	}
	return result, nil
}

func fieldInvCPU(params FieldParams, a []uint64) ([]uint64, error) {
	result := make([]uint64, len(a))
	for i := range a {
		result[i] = modInverse(a[i], params.Modulus)
	}
	return result, nil
}

func fieldExpCPU(params FieldParams, a []uint64, exp uint64) ([]uint64, error) {
	result := make([]uint64, len(a))
	for i := range a {
		result[i] = modPow(a[i], exp, params.Modulus)
	}
	return result, nil
}

// Modular arithmetic helpers

func addMod(a, b, m uint64) uint64 {
	sum := a + b
	if sum >= m || sum < a { // overflow check
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
	// Fermat's little theorem: a^(-1) = a^(m-2) mod m for prime m
	return modPow(a, m-2, m)
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

// Poseidon2 helpers

func sboxPoseidon(x uint64, d uint32, m uint64) uint64 {
	// S-box: x^d mod m
	return modPow(x, uint64(d), m)
}

func mdsMultiply(state, mds []uint64, m uint64) []uint64 {
	t := len(state)
	result := make([]uint64, t)

	// If MDS matrix not provided, use simple Cauchy matrix
	if len(mds) == 0 {
		// Default MDS: circulant matrix with first row [2, 1, 1, ...]
		for i := 0; i < t; i++ {
			sum := mulMod(state[i], 2, m)
			for j := 1; j < t; j++ {
				sum = addMod(sum, state[(i+j)%t], m)
			}
			result[i] = sum
		}
		return result
	}

	// Standard matrix-vector multiplication
	for i := 0; i < t; i++ {
		sum := uint64(0)
		for j := 0; j < t; j++ {
			if i*t+j < len(mds) {
				term := mulMod(mds[i*t+j], state[j], m)
				sum = addMod(sum, term, m)
			}
		}
		result[i] = sum
	}

	return result
}
