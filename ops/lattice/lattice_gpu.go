//go:build accel

package lattice

import (
	"github.com/luxfi/accel"
)

// GPU-accelerated lattice cryptography implementations.
// These dispatch to the accel library's GPU backends (CUDA, Metal, Vulkan)
// for parallel NTT, polynomial arithmetic, and sampling operations.

// computeNTTRoots precomputes roots of unity for NTT
func computeNTTRoots(params NTTParams) []uint64 {
	n := int(params.N)
	roots := make([]uint64, n)
	roots[0] = 1
	for i := 1; i < n; i++ {
		roots[i] = mulMod(roots[i-1], params.Root, params.Modulus)
	}
	return roots
}

// computeINTTRoots precomputes inverse roots for INTT
func computeINTTRoots(params NTTParams) []uint64 {
	n := int(params.N)
	invRoot := modInverse(params.Root, params.Modulus)
	roots := make([]uint64, n)
	roots[0] = 1
	for i := 1; i < n; i++ {
		roots[i] = mulMod(roots[i-1], invRoot, params.Modulus)
	}
	return roots
}

func nttForwardGPU(sess *accel.Session, params NTTParams, coeffs []uint64) ([]uint64, error) {
	n := int(params.N)

	// Create input tensor
	inputTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, coeffs)
	if err != nil {
		return nttForwardCPU(params, coeffs)
	}
	defer inputTensor.Close()

	// Create output tensor
	outputTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return nttForwardCPU(params, coeffs)
	}
	defer outputTensor.Close()

	// Precompute roots of unity
	roots := computeNTTRoots(params)
	rootsTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, roots)
	if err != nil {
		return nttForwardCPU(params, coeffs)
	}
	defer rootsTensor.Close()

	// Dispatch NTT kernel via ZK ops
	err = sess.ZK().NTT(
		inputTensor.Untyped(),
		outputTensor.Untyped(),
		rootsTensor.Untyped(),
		params.Modulus,
	)
	if err != nil {
		return nttForwardCPU(params, coeffs)
	}

	// Sync
	if err := sess.Sync(); err != nil {
		return nttForwardCPU(params, coeffs)
	}

	// Read results
	result, err := outputTensor.ToSlice()
	if err != nil {
		return nttForwardCPU(params, coeffs)
	}

	return result, nil
}

func nttInverseGPU(sess *accel.Session, params NTTParams, evals []uint64) ([]uint64, error) {
	n := int(params.N)

	// Create input tensor
	inputTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, evals)
	if err != nil {
		return nttInverseCPU(params, evals)
	}
	defer inputTensor.Close()

	// Create output tensor
	outputTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return nttInverseCPU(params, evals)
	}
	defer outputTensor.Close()

	// Precompute inverse roots
	invRoots := computeINTTRoots(params)
	invRootsTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, invRoots)
	if err != nil {
		return nttInverseCPU(params, evals)
	}
	defer invRootsTensor.Close()

	// Dispatch INTT kernel via ZK ops
	err = sess.ZK().INTT(
		inputTensor.Untyped(),
		outputTensor.Untyped(),
		invRootsTensor.Untyped(),
		params.Modulus,
	)
	if err != nil {
		return nttInverseCPU(params, evals)
	}

	// Sync
	if err := sess.Sync(); err != nil {
		return nttInverseCPU(params, evals)
	}

	// Read results
	result, err := outputTensor.ToSlice()
	if err != nil {
		return nttInverseCPU(params, evals)
	}

	return result, nil
}

func polyMulGPU(sess *accel.Session, params NTTParams, a, b []uint64) ([]uint64, error) {
	n := int(params.N)

	// Create tensors
	aTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, a)
	if err != nil {
		return polyMulCPU(params, a, b)
	}
	defer aTensor.Close()

	bTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, b)
	if err != nil {
		return polyMulCPU(params, a, b)
	}
	defer bTensor.Close()

	cTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return polyMulCPU(params, a, b)
	}
	defer cTensor.Close()

	// Dispatch polynomial multiplication kernel
	// This uses NTT internally: c = INTT(NTT(a) * NTT(b))
	err = sess.ZK().PolyMul(
		aTensor.Untyped(),
		bTensor.Untyped(),
		cTensor.Untyped(),
		params.Modulus,
	)
	if err != nil {
		return polyMulCPU(params, a, b)
	}

	if err := sess.Sync(); err != nil {
		return polyMulCPU(params, a, b)
	}

	result, err := cTensor.ToSlice()
	if err != nil {
		return polyMulCPU(params, a, b)
	}

	return result, nil
}

func polyAddGPU(sess *accel.Session, modulus uint64, a, b []uint64) ([]uint64, error) {
	n := len(a)

	// Create tensors
	aTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, a)
	if err != nil {
		return polyAddCPU(modulus, a, b)
	}
	defer aTensor.Close()

	bTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, b)
	if err != nil {
		return polyAddCPU(modulus, a, b)
	}
	defer bTensor.Close()

	cTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return polyAddCPU(modulus, a, b)
	}
	defer cTensor.Close()

	// Dispatch field add kernel
	err = sess.ZK().FieldAdd(
		aTensor.Untyped(),
		bTensor.Untyped(),
		cTensor.Untyped(),
		modulus,
	)
	if err != nil {
		return polyAddCPU(modulus, a, b)
	}

	if err := sess.Sync(); err != nil {
		return polyAddCPU(modulus, a, b)
	}

	result, err := cTensor.ToSlice()
	if err != nil {
		return polyAddCPU(modulus, a, b)
	}

	return result, nil
}

func polySubGPU(sess *accel.Session, modulus uint64, a, b []uint64) ([]uint64, error) {
	n := len(a)

	// Create tensors
	aTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, a)
	if err != nil {
		return polySubCPU(modulus, a, b)
	}
	defer aTensor.Close()

	// Compute -b = modulus - b for each element
	negB := make([]uint64, n)
	for i := range b {
		if b[i] == 0 {
			negB[i] = 0
		} else {
			negB[i] = modulus - b[i]
		}
	}

	negBTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, negB)
	if err != nil {
		return polySubCPU(modulus, a, b)
	}
	defer negBTensor.Close()

	cTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return polySubCPU(modulus, a, b)
	}
	defer cTensor.Close()

	// a + (-b) = a - b
	err = sess.ZK().FieldAdd(
		aTensor.Untyped(),
		negBTensor.Untyped(),
		cTensor.Untyped(),
		modulus,
	)
	if err != nil {
		return polySubCPU(modulus, a, b)
	}

	if err := sess.Sync(); err != nil {
		return polySubCPU(modulus, a, b)
	}

	result, err := cTensor.ToSlice()
	if err != nil {
		return polySubCPU(modulus, a, b)
	}

	return result, nil
}

func batchNTTForwardGPU(sess *accel.Session, params NTTParams, polys [][]uint64) ([][]uint64, error) {
	m := len(polys)
	n := int(params.N)

	// Precompute roots
	roots := computeNTTRoots(params)
	rootsTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, roots)
	if err != nil {
		return batchNTTForwardCPU(params, polys)
	}
	defer rootsTensor.Close()

	// Process each polynomial
	results := make([][]uint64, m)
	for i := 0; i < m; i++ {
		polyInput, err := accel.NewTensorWithData[uint64](sess, []int{n}, polys[i])
		if err != nil {
			return batchNTTForwardCPU(params, polys)
		}

		polyOutput, err := accel.NewTensor[uint64](sess, []int{n})
		if err != nil {
			polyInput.Close()
			return batchNTTForwardCPU(params, polys)
		}

		err = sess.ZK().NTT(
			polyInput.Untyped(),
			polyOutput.Untyped(),
			rootsTensor.Untyped(),
			params.Modulus,
		)

		if err != nil {
			polyInput.Close()
			polyOutput.Close()
			return batchNTTForwardCPU(params, polys)
		}

		if err := sess.Sync(); err != nil {
			polyInput.Close()
			polyOutput.Close()
			return batchNTTForwardCPU(params, polys)
		}

		result, err := polyOutput.ToSlice()
		if err != nil {
			polyInput.Close()
			polyOutput.Close()
			return batchNTTForwardCPU(params, polys)
		}

		results[i] = result
		polyInput.Close()
		polyOutput.Close()
	}

	return results, nil
}

func batchNTTInverseGPU(sess *accel.Session, params NTTParams, polys [][]uint64) ([][]uint64, error) {
	m := len(polys)
	n := int(params.N)

	// Precompute inverse roots
	invRoots := computeINTTRoots(params)
	invRootsTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, invRoots)
	if err != nil {
		return batchNTTInverseCPU(params, polys)
	}
	defer invRootsTensor.Close()

	// Process each polynomial
	results := make([][]uint64, m)
	for i := 0; i < m; i++ {
		polyInput, err := accel.NewTensorWithData[uint64](sess, []int{n}, polys[i])
		if err != nil {
			return batchNTTInverseCPU(params, polys)
		}

		polyOutput, err := accel.NewTensor[uint64](sess, []int{n})
		if err != nil {
			polyInput.Close()
			return batchNTTInverseCPU(params, polys)
		}

		err = sess.ZK().INTT(
			polyInput.Untyped(),
			polyOutput.Untyped(),
			invRootsTensor.Untyped(),
			params.Modulus,
		)

		if err != nil {
			polyInput.Close()
			polyOutput.Close()
			return batchNTTInverseCPU(params, polys)
		}

		if err := sess.Sync(); err != nil {
			polyInput.Close()
			polyOutput.Close()
			return batchNTTInverseCPU(params, polys)
		}

		result, err := polyOutput.ToSlice()
		if err != nil {
			polyInput.Close()
			polyOutput.Close()
			return batchNTTInverseCPU(params, polys)
		}

		results[i] = result
		polyInput.Close()
		polyOutput.Close()
	}

	return results, nil
}

func sampleNTTGPU(sess *accel.Session, params NTTParams, seed []byte) ([]uint64, error) {
	n := int(params.N)

	// Create seed tensor
	seedTensor, err := accel.NewTensorWithData[uint8](sess, []int{len(seed)}, seed)
	if err != nil {
		return sampleNTTCPU(params, seed)
	}
	defer seedTensor.Close()

	// Create output tensor for hash expansion
	// We need n * 32 bytes for n hash outputs
	hashOutputTensor, err := accel.NewTensor[uint8](sess, []int{n, 32})
	if err != nil {
		return sampleNTTCPU(params, seed)
	}
	defer hashOutputTensor.Close()

	// Use crypto SHA256 for hash expansion
	// Note: This is a simplified approach - real implementation would use
	// a proper hash-to-field function
	err = sess.Crypto().SHA256(seedTensor.Untyped(), hashOutputTensor.Untyped())
	if err != nil {
		return sampleNTTCPU(params, seed)
	}

	if err := sess.Sync(); err != nil {
		return sampleNTTCPU(params, seed)
	}

	// For proper sampling with counter mode, fall back to CPU
	// GPU acceleration is more beneficial for large polynomial operations
	return sampleNTTCPU(params, seed)
}
