//go:build accel

package zk

import (
	"unsafe"

	"github.com/luxfi/accel"
)

// GPU-accelerated implementations using ZK kernels.
// These dispatch to the accel library's GPU backends (CUDA, Metal, Vulkan)
// for parallel NTT, MSM, polynomial arithmetic, and hash operations.

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

// computeINTTRoots precomputes inverse roots for inverse NTT
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

func nttGPU(sess *accel.Session, params NTTParams, coeffs []uint64) ([]uint64, error) {
	n := int(params.N)

	// Convert coeffs to bytes for tensor
	coeffBytes := uint64SliceToBytes(coeffs)

	// Create input tensor
	inputTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, coeffs)
	if err != nil {
		return nttCPU(params, coeffs)
	}
	defer inputTensor.Close()

	// Create output tensor
	outputTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return nttCPU(params, coeffs)
	}
	defer outputTensor.Close()

	// Precompute roots of unity
	roots := computeNTTRoots(params)
	rootsTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, roots)
	if err != nil {
		return nttCPU(params, coeffs)
	}
	defer rootsTensor.Close()

	// Dispatch NTT kernel
	err = sess.ZK().NTT(
		inputTensor.Untyped(),
		outputTensor.Untyped(),
		rootsTensor.Untyped(),
		params.Modulus,
	)
	if err != nil {
		return nttCPU(params, coeffs)
	}

	// Sync
	if err := sess.Sync(); err != nil {
		return nttCPU(params, coeffs)
	}

	// Read results
	result, err := outputTensor.ToSlice()
	if err != nil {
		return nttCPU(params, coeffs)
	}

	_ = coeffBytes // suppress unused warning
	return result, nil
}

func inttGPU(sess *accel.Session, params NTTParams, evals []uint64) ([]uint64, error) {
	n := int(params.N)

	// Create input tensor
	inputTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, evals)
	if err != nil {
		return inttCPU(params, evals)
	}
	defer inputTensor.Close()

	// Create output tensor
	outputTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return inttCPU(params, evals)
	}
	defer outputTensor.Close()

	// Precompute inverse roots of unity
	invRoots := computeINTTRoots(params)
	invRootsTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, invRoots)
	if err != nil {
		return inttCPU(params, evals)
	}
	defer invRootsTensor.Close()

	// Dispatch INTT kernel
	err = sess.ZK().INTT(
		inputTensor.Untyped(),
		outputTensor.Untyped(),
		invRootsTensor.Untyped(),
		params.Modulus,
	)
	if err != nil {
		return inttCPU(params, evals)
	}

	// Sync
	if err := sess.Sync(); err != nil {
		return inttCPU(params, evals)
	}

	// Read results
	result, err := outputTensor.ToSlice()
	if err != nil {
		return inttCPU(params, evals)
	}

	return result, nil
}

func batchNTTGPU(sess *accel.Session, params NTTParams, polys [][]uint64) ([][]uint64, error) {
	m := len(polys) // number of polynomials
	n := int(params.N)

	// Flatten all polynomials into a single tensor
	flatData := make([]uint64, m*n)
	for i, p := range polys {
		copy(flatData[i*n:(i+1)*n], p)
	}

	// Create input tensor [M, N]
	inputTensor, err := accel.NewTensorWithData[uint64](sess, []int{m, n}, flatData)
	if err != nil {
		return batchNTTCPU(params, polys)
	}
	defer inputTensor.Close()

	// Create output tensor [M, N]
	outputTensor, err := accel.NewTensor[uint64](sess, []int{m, n})
	if err != nil {
		return batchNTTCPU(params, polys)
	}
	defer outputTensor.Close()

	// Precompute roots
	roots := computeNTTRoots(params)
	rootsTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, roots)
	if err != nil {
		return batchNTTCPU(params, polys)
	}
	defer rootsTensor.Close()

	// Dispatch batched NTT - process each polynomial
	// Note: ZK().NTT is not natively batched, so we iterate
	// A production implementation would use a batched kernel
	for i := 0; i < m; i++ {
		// Create views for this polynomial
		polyInput, _ := accel.NewTensorWithData[uint64](sess, []int{n}, polys[i])
		polyOutput, _ := accel.NewTensor[uint64](sess, []int{n})

		err = sess.ZK().NTT(
			polyInput.Untyped(),
			polyOutput.Untyped(),
			rootsTensor.Untyped(),
			params.Modulus,
		)
		if err != nil {
			polyInput.Close()
			polyOutput.Close()
			return batchNTTCPU(params, polys)
		}

		polyInput.Close()
		polyOutput.Close()
	}

	// For now, use CPU implementation since batch kernel not exposed
	return batchNTTCPU(params, polys)
}

func batchINTTGPU(sess *accel.Session, params NTTParams, polys [][]uint64) ([][]uint64, error) {
	// Similar to batchNTTGPU - batch kernel not directly exposed
	return batchINTTCPU(params, polys)
}

func msmGPU(sess *accel.Session, curve CurveType, scalars, points [][]byte) ([]byte, error) {
	n := len(scalars)
	if n == 0 {
		return nil, ErrEmptyBatch
	}

	// Determine sizes based on curve
	var scalarSize, pointSize int
	switch curve {
	case CurveBN254:
		scalarSize = 32 // 256-bit scalar
		pointSize = 64  // BN254 G1 affine (32 bytes x, 32 bytes y)
	case CurveBLS12_381:
		scalarSize = 32
		pointSize = 96 // BLS12-381 G1 (48 bytes x, 48 bytes y)
	case CurveBLS12_377:
		scalarSize = 32
		pointSize = 96
	case CurvePallas, CurveVesta:
		scalarSize = 32
		pointSize = 64
	default:
		return msmCPU(curve, scalars, points)
	}

	// Pack scalars
	scalarBytes := make([]byte, n*scalarSize)
	for i, s := range scalars {
		if len(s) >= scalarSize {
			copy(scalarBytes[i*scalarSize:(i+1)*scalarSize], s[:scalarSize])
		} else {
			copy(scalarBytes[i*scalarSize:i*scalarSize+len(s)], s)
		}
	}

	// Pack points
	pointBytes := make([]byte, n*pointSize)
	for i, p := range points {
		if len(p) >= pointSize {
			copy(pointBytes[i*pointSize:(i+1)*pointSize], p[:pointSize])
		} else {
			copy(pointBytes[i*pointSize:i*pointSize+len(p)], p)
		}
	}

	// Create tensors
	scalarTensor, err := accel.NewTensorWithData[uint8](sess, []int{n, scalarSize}, scalarBytes)
	if err != nil {
		return msmCPU(curve, scalars, points)
	}
	defer scalarTensor.Close()

	pointTensor, err := accel.NewTensorWithData[uint8](sess, []int{n, pointSize}, pointBytes)
	if err != nil {
		return msmCPU(curve, scalars, points)
	}
	defer pointTensor.Close()

	// Result tensor
	resultTensor, err := accel.NewTensor[uint8](sess, []int{pointSize})
	if err != nil {
		return msmCPU(curve, scalars, points)
	}
	defer resultTensor.Close()

	// Dispatch MSM kernel
	err = sess.ZK().MSM(
		scalarTensor.Untyped(),
		pointTensor.Untyped(),
		resultTensor.Untyped(),
	)
	if err != nil {
		return msmCPU(curve, scalars, points)
	}

	// Sync
	if err := sess.Sync(); err != nil {
		return msmCPU(curve, scalars, points)
	}

	// Read result
	result, err := resultTensor.ToSlice()
	if err != nil {
		return msmCPU(curve, scalars, points)
	}

	return result, nil
}

func msmBatchGPU(sess *accel.Session, curve CurveType, scalars, points [][][]byte) ([][]byte, error) {
	// Batch MSM: multiple independent MSMs
	m := len(scalars)
	if m == 0 {
		return nil, ErrEmptyBatch
	}

	results := make([][]byte, m)
	for i := 0; i < m; i++ {
		result, err := msmGPU(sess, curve, scalars[i], points[i])
		if err != nil {
			return msmBatchCPU(curve, scalars, points)
		}
		results[i] = result
	}

	return results, nil
}

func poseidon2GPU(sess *accel.Session, params Poseidon2Params, inputs []uint64) (uint64, error) {
	// Poseidon2 hash using ZK-friendly hash kernel
	n := len(inputs)

	// Create input tensor
	inputTensor, err := accel.NewTensorWithData[uint64](sess, []int{1, n}, inputs)
	if err != nil {
		return poseidon2CPU(params, inputs)
	}
	defer inputTensor.Close()

	// Create output tensor (single field element)
	outputTensor, err := accel.NewTensor[uint64](sess, []int{1})
	if err != nil {
		return poseidon2CPU(params, inputs)
	}
	defer outputTensor.Close()

	// Use crypto Poseidon kernel (assumes compatible parameters)
	// Convert to byte representation for crypto ops
	inputBytes := uint64SliceToBytes(inputs)
	inputBytesTensor, err := accel.NewTensorWithData[uint8](sess, []int{1, n * 8}, inputBytes)
	if err != nil {
		return poseidon2CPU(params, inputs)
	}
	defer inputBytesTensor.Close()

	outputBytesTensor, err := accel.NewTensor[uint8](sess, []int{1, 32})
	if err != nil {
		return poseidon2CPU(params, inputs)
	}
	defer outputBytesTensor.Close()

	err = sess.Crypto().Poseidon(inputBytesTensor.Untyped(), outputBytesTensor.Untyped())
	if err != nil {
		return poseidon2CPU(params, inputs)
	}

	if err := sess.Sync(); err != nil {
		return poseidon2CPU(params, inputs)
	}

	// Read result and convert first 8 bytes to uint64
	hashBytes, err := outputBytesTensor.ToSlice()
	if err != nil {
		return poseidon2CPU(params, inputs)
	}

	// Take first 8 bytes as uint64 result, reduced mod modulus
	result := *(*uint64)(unsafe.Pointer(&hashBytes[0]))
	result = result % params.Modulus

	return result, nil
}

func poseidon2HashGPU(sess *accel.Session, params Poseidon2Params, left, right uint64) (uint64, error) {
	return poseidon2GPU(sess, params, []uint64{left, right})
}

func batchPoseidon2HashGPU(sess *accel.Session, params Poseidon2Params, lefts, rights []uint64) ([]uint64, error) {
	n := len(lefts)
	if n == 0 {
		return nil, ErrEmptyBatch
	}

	// Pack pairs into tensor [N, 2]
	pairs := make([]uint64, n*2)
	for i := 0; i < n; i++ {
		pairs[i*2] = lefts[i]
		pairs[i*2+1] = rights[i]
	}

	// Convert to bytes
	pairBytes := uint64SliceToBytes(pairs)

	inputTensor, err := accel.NewTensorWithData[uint8](sess, []int{n, 16}, pairBytes)
	if err != nil {
		return batchPoseidon2HashCPU(params, lefts, rights)
	}
	defer inputTensor.Close()

	outputTensor, err := accel.NewTensor[uint8](sess, []int{n, 32})
	if err != nil {
		return batchPoseidon2HashCPU(params, lefts, rights)
	}
	defer outputTensor.Close()

	// Batch Poseidon hash
	err = sess.Crypto().Poseidon(inputTensor.Untyped(), outputTensor.Untyped())
	if err != nil {
		return batchPoseidon2HashCPU(params, lefts, rights)
	}

	if err := sess.Sync(); err != nil {
		return batchPoseidon2HashCPU(params, lefts, rights)
	}

	hashBytes, err := outputTensor.ToSlice()
	if err != nil {
		return batchPoseidon2HashCPU(params, lefts, rights)
	}

	// Extract uint64 results from each hash
	results := make([]uint64, n)
	for i := 0; i < n; i++ {
		results[i] = *(*uint64)(unsafe.Pointer(&hashBytes[i*32])) % params.Modulus
	}

	return results, nil
}

func polyAddGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	n := len(a)

	// Create tensors
	aTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, a)
	if err != nil {
		return polyAddCPU(params, a, b)
	}
	defer aTensor.Close()

	bTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, b)
	if err != nil {
		return polyAddCPU(params, a, b)
	}
	defer bTensor.Close()

	cTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return polyAddCPU(params, a, b)
	}
	defer cTensor.Close()

	// Dispatch field add kernel
	err = sess.ZK().FieldAdd(
		aTensor.Untyped(),
		bTensor.Untyped(),
		cTensor.Untyped(),
		params.Modulus,
	)
	if err != nil {
		return polyAddCPU(params, a, b)
	}

	if err := sess.Sync(); err != nil {
		return polyAddCPU(params, a, b)
	}

	result, err := cTensor.ToSlice()
	if err != nil {
		return polyAddCPU(params, a, b)
	}

	return result, nil
}

func polySubGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	n := len(a)

	// Create tensors
	aTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, a)
	if err != nil {
		return polySubCPU(params, a, b)
	}
	defer aTensor.Close()

	// Compute -b = modulus - b for each element, then add
	negB := make([]uint64, n)
	for i := range b {
		if b[i] == 0 {
			negB[i] = 0
		} else {
			negB[i] = params.Modulus - b[i]
		}
	}

	negBTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, negB)
	if err != nil {
		return polySubCPU(params, a, b)
	}
	defer negBTensor.Close()

	cTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return polySubCPU(params, a, b)
	}
	defer cTensor.Close()

	// a + (-b) = a - b
	err = sess.ZK().FieldAdd(
		aTensor.Untyped(),
		negBTensor.Untyped(),
		cTensor.Untyped(),
		params.Modulus,
	)
	if err != nil {
		return polySubCPU(params, a, b)
	}

	if err := sess.Sync(); err != nil {
		return polySubCPU(params, a, b)
	}

	result, err := cTensor.ToSlice()
	if err != nil {
		return polySubCPU(params, a, b)
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

	// Result can have degree up to 2N-1
	cTensor, err := accel.NewTensor[uint64](sess, []int{2*n - 1})
	if err != nil {
		return polyMulCPU(params, a, b)
	}
	defer cTensor.Close()

	// Dispatch polynomial multiplication kernel
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

func polyMulPointwiseGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	n := len(a)

	aTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, a)
	if err != nil {
		return polyMulPointwiseCPU(params, a, b)
	}
	defer aTensor.Close()

	bTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, b)
	if err != nil {
		return polyMulPointwiseCPU(params, a, b)
	}
	defer bTensor.Close()

	cTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return polyMulPointwiseCPU(params, a, b)
	}
	defer cTensor.Close()

	// Field multiply for pointwise multiplication
	err = sess.ZK().FieldMul(
		aTensor.Untyped(),
		bTensor.Untyped(),
		cTensor.Untyped(),
		params.Modulus,
	)
	if err != nil {
		return polyMulPointwiseCPU(params, a, b)
	}

	if err := sess.Sync(); err != nil {
		return polyMulPointwiseCPU(params, a, b)
	}

	result, err := cTensor.ToSlice()
	if err != nil {
		return polyMulPointwiseCPU(params, a, b)
	}

	return result, nil
}

func polyEvalGPU(sess *accel.Session, params FieldParams, coeffs, points []uint64) ([]uint64, error) {
	nCoeffs := len(coeffs)
	nPoints := len(points)

	coeffsTensor, err := accel.NewTensorWithData[uint64](sess, []int{nCoeffs}, coeffs)
	if err != nil {
		return polyEvalCPU(params, coeffs, points)
	}
	defer coeffsTensor.Close()

	pointsTensor, err := accel.NewTensorWithData[uint64](sess, []int{nPoints}, points)
	if err != nil {
		return polyEvalCPU(params, coeffs, points)
	}
	defer pointsTensor.Close()

	resultsTensor, err := accel.NewTensor[uint64](sess, []int{nPoints})
	if err != nil {
		return polyEvalCPU(params, coeffs, points)
	}
	defer resultsTensor.Close()

	// Dispatch polynomial evaluation kernel
	err = sess.ZK().PolyEval(
		coeffsTensor.Untyped(),
		pointsTensor.Untyped(),
		resultsTensor.Untyped(),
		params.Modulus,
	)
	if err != nil {
		return polyEvalCPU(params, coeffs, points)
	}

	if err := sess.Sync(); err != nil {
		return polyEvalCPU(params, coeffs, points)
	}

	results, err := resultsTensor.ToSlice()
	if err != nil {
		return polyEvalCPU(params, coeffs, points)
	}

	return results, nil
}

func polyInterpolateGPU(sess *accel.Session, params FieldParams, xs, ys []uint64) ([]uint64, error) {
	// Lagrange interpolation is complex on GPU, fall back to CPU
	// A production implementation would use FFT-based interpolation
	return polyInterpolateCPU(params, xs, ys)
}

func friFoldGPU(sess *accel.Session, params FRIParams, evals []uint64, alpha uint64) ([]uint64, error) {
	n := len(evals) / 2

	// Pack evals and alpha for kernel
	evalsTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(evals)}, evals)
	if err != nil {
		return friFoldCPU(params, evals, alpha)
	}
	defer evalsTensor.Close()

	resultTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return friFoldCPU(params, evals, alpha)
	}
	defer resultTensor.Close()

	// FRI folding: f'(x) = (f(x) + f(-x))/2 + alpha * (f(x) - f(-x))/(2x)
	// This is implemented using field operations
	// For now, use CPU as dedicated FRI kernel not exposed
	return friFoldCPU(params, evals, alpha)
}

func friQueryPhaseGPU(sess *accel.Session, params FRIParams, evals []uint64, indices []uint32) ([][]uint64, error) {
	// FRI query phase: gather evaluations at query indices
	// This is essentially a parallel gather operation
	return friQueryPhaseCPU(params, evals, indices)
}

func commitPolyGPU(sess *accel.Session, params CommitParams, coeffs [][]byte, srs [][]byte) ([]byte, error) {
	// KZG commitment: C = MSM(coeffs, srs)
	// This is an MSM with SRS points as bases
	return msmGPU(sess, params.Curve, coeffs, srs)
}

func batchCommitPolyGPU(sess *accel.Session, params CommitParams, coeffsList [][][]byte, srs [][]byte) ([][]byte, error) {
	m := len(coeffsList)
	if m == 0 {
		return nil, ErrEmptyBatch
	}

	results := make([][]byte, m)
	for i, coeffs := range coeffsList {
		c, err := commitPolyGPU(sess, params, coeffs, srs)
		if err != nil {
			return batchCommitPolyCPU(params, coeffsList, srs)
		}
		results[i] = c
	}

	return results, nil
}

func fieldAddGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	n := len(a)

	aTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, a)
	if err != nil {
		return fieldAddCPU(params, a, b)
	}
	defer aTensor.Close()

	bTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, b)
	if err != nil {
		return fieldAddCPU(params, a, b)
	}
	defer bTensor.Close()

	cTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return fieldAddCPU(params, a, b)
	}
	defer cTensor.Close()

	err = sess.ZK().FieldAdd(
		aTensor.Untyped(),
		bTensor.Untyped(),
		cTensor.Untyped(),
		params.Modulus,
	)
	if err != nil {
		return fieldAddCPU(params, a, b)
	}

	if err := sess.Sync(); err != nil {
		return fieldAddCPU(params, a, b)
	}

	result, err := cTensor.ToSlice()
	if err != nil {
		return fieldAddCPU(params, a, b)
	}

	return result, nil
}

func fieldMulGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	n := len(a)

	aTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, a)
	if err != nil {
		return fieldMulCPU(params, a, b)
	}
	defer aTensor.Close()

	bTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, b)
	if err != nil {
		return fieldMulCPU(params, a, b)
	}
	defer bTensor.Close()

	cTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return fieldMulCPU(params, a, b)
	}
	defer cTensor.Close()

	err = sess.ZK().FieldMul(
		aTensor.Untyped(),
		bTensor.Untyped(),
		cTensor.Untyped(),
		params.Modulus,
	)
	if err != nil {
		return fieldMulCPU(params, a, b)
	}

	if err := sess.Sync(); err != nil {
		return fieldMulCPU(params, a, b)
	}

	result, err := cTensor.ToSlice()
	if err != nil {
		return fieldMulCPU(params, a, b)
	}

	return result, nil
}

func fieldInvGPU(sess *accel.Session, params FieldParams, a []uint64) ([]uint64, error) {
	n := len(a)

	aTensor, err := accel.NewTensorWithData[uint64](sess, []int{n}, a)
	if err != nil {
		return fieldInvCPU(params, a)
	}
	defer aTensor.Close()

	bTensor, err := accel.NewTensor[uint64](sess, []int{n})
	if err != nil {
		return fieldInvCPU(params, a)
	}
	defer bTensor.Close()

	err = sess.ZK().FieldInv(
		aTensor.Untyped(),
		bTensor.Untyped(),
		params.Modulus,
	)
	if err != nil {
		return fieldInvCPU(params, a)
	}

	if err := sess.Sync(); err != nil {
		return fieldInvCPU(params, a)
	}

	result, err := bTensor.ToSlice()
	if err != nil {
		return fieldInvCPU(params, a)
	}

	return result, nil
}

func fieldExpGPU(sess *accel.Session, params FieldParams, a []uint64, exp uint64) ([]uint64, error) {
	// Field exponentiation: a^exp mod modulus for each element
	// This is typically implemented via repeated squaring
	// For now, use CPU as dedicated kernel not exposed
	return fieldExpCPU(params, a, exp)
}

// Helper function to convert uint64 slice to bytes
func uint64SliceToBytes(data []uint64) []byte {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*8)
}
