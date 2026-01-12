//go:build accel

package zk

import (
	"github.com/luxfi/accel"
)

// GPU-accelerated implementations using ZK kernels.
// These dispatch to the accel library's GPU backends (CUDA, Metal, Vulkan).

func nttGPU(sess *accel.Session, params NTTParams, coeffs []uint64) ([]uint64, error) {
	// Uses zk/ntt kernel for GPU-accelerated NTT
	// TODO: Implement kernel dispatch via Session.ZK().NTT()
	return nttCPU(params, coeffs)
}

func inttGPU(sess *accel.Session, params NTTParams, evals []uint64) ([]uint64, error) {
	// Uses zk/intt kernel for inverse NTT
	// TODO: Implement kernel dispatch
	return inttCPU(params, evals)
}

func batchNTTGPU(sess *accel.Session, params NTTParams, polys [][]uint64) ([][]uint64, error) {
	// Uses zk/ntt_batch kernel for parallel NTT
	// TODO: Implement kernel dispatch
	return batchNTTCPU(params, polys)
}

func batchINTTGPU(sess *accel.Session, params NTTParams, polys [][]uint64) ([][]uint64, error) {
	// Uses zk/intt_batch kernel for parallel inverse NTT
	// TODO: Implement kernel dispatch
	return batchINTTCPU(params, polys)
}

func msmGPU(sess *accel.Session, curve CurveType, scalars, points [][]byte) ([]byte, error) {
	// Uses zk/msm kernel with curve-specific sub-kernel
	// MSM is typically the most GPU-intensive ZK operation
	//
	// Kernel selection by curve:
	// - CurveBN254:     zk/msm_bn254
	// - CurveBLS12_381: zk/msm_bls12_381
	// - CurveBLS12_377: zk/msm_bls12_377
	// - CurvePallas:    zk/msm_pallas
	// - CurveVesta:     zk/msm_vesta
	//
	// TODO: Implement kernel dispatch
	return msmCPU(curve, scalars, points)
}

func msmBatchGPU(sess *accel.Session, curve CurveType, scalars, points [][][]byte) ([][]byte, error) {
	// Uses zk/msm_batch kernel for multiple MSMs
	// TODO: Implement kernel dispatch
	return msmBatchCPU(curve, scalars, points)
}

func poseidon2GPU(sess *accel.Session, params Poseidon2Params, inputs []uint64) (uint64, error) {
	// Uses zk/poseidon2 kernel
	// Poseidon2 is optimized for ZK circuits and Merkle trees
	// TODO: Implement kernel dispatch
	return poseidon2CPU(params, inputs)
}

func poseidon2HashGPU(sess *accel.Session, params Poseidon2Params, left, right uint64) (uint64, error) {
	// Uses zk/poseidon2_merkle kernel optimized for 2-to-1 hashing
	// TODO: Implement kernel dispatch
	return poseidon2HashCPU(params, left, right)
}

func batchPoseidon2HashGPU(sess *accel.Session, params Poseidon2Params, lefts, rights []uint64) ([]uint64, error) {
	// Uses zk/poseidon2_batch kernel for parallel Merkle tree construction
	// This is critical for building Merkle trees in ZK provers
	// TODO: Implement kernel dispatch
	return batchPoseidon2HashCPU(params, lefts, rights)
}

func polyAddGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	// Uses zk/poly_add kernel
	// TODO: Implement kernel dispatch
	return polyAddCPU(params, a, b)
}

func polySubGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	// Uses zk/poly_sub kernel
	// TODO: Implement kernel dispatch
	return polySubCPU(params, a, b)
}

func polyMulGPU(sess *accel.Session, params NTTParams, a, b []uint64) ([]uint64, error) {
	// Uses zk/poly_mul kernel (NTT-based multiplication)
	// TODO: Implement kernel dispatch
	return polyMulCPU(params, a, b)
}

func polyMulPointwiseGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	// Uses zk/poly_mul_pointwise kernel
	// TODO: Implement kernel dispatch
	return polyMulPointwiseCPU(params, a, b)
}

func polyEvalGPU(sess *accel.Session, params FieldParams, coeffs, points []uint64) ([]uint64, error) {
	// Uses zk/poly_eval kernel
	// GPU acceleration significant for large degree polynomials
	// TODO: Implement kernel dispatch
	return polyEvalCPU(params, coeffs, points)
}

func polyInterpolateGPU(sess *accel.Session, params FieldParams, xs, ys []uint64) ([]uint64, error) {
	// Uses zk/poly_interpolate kernel
	// TODO: Implement kernel dispatch
	return polyInterpolateCPU(params, xs, ys)
}

func friFoldGPU(sess *accel.Session, params FRIParams, evals []uint64, alpha uint64) ([]uint64, error) {
	// Uses zk/fri_fold kernel
	// FRI folding is critical for STARK provers
	// TODO: Implement kernel dispatch
	return friFoldCPU(params, evals, alpha)
}

func friQueryPhaseGPU(sess *accel.Session, params FRIParams, evals []uint64, indices []uint32) ([][]uint64, error) {
	// Uses zk/fri_query kernel
	// TODO: Implement kernel dispatch
	return friQueryPhaseCPU(params, evals, indices)
}

func commitPolyGPU(sess *accel.Session, params CommitParams, coeffs [][]byte, srs [][]byte) ([]byte, error) {
	// Uses zk/kzg_commit kernel
	// Polynomial commitment is MSM with SRS points
	// TODO: Implement kernel dispatch
	return commitPolyCPU(params, coeffs, srs)
}

func batchCommitPolyGPU(sess *accel.Session, params CommitParams, coeffsList [][][]byte, srs [][]byte) ([][]byte, error) {
	// Uses zk/kzg_commit_batch kernel
	// TODO: Implement kernel dispatch
	return batchCommitPolyCPU(params, coeffsList, srs)
}

func fieldAddGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	// Uses zk/field_add kernel
	// TODO: Implement kernel dispatch
	return fieldAddCPU(params, a, b)
}

func fieldMulGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	// Uses zk/field_mul kernel with Montgomery multiplication
	// TODO: Implement kernel dispatch
	return fieldMulCPU(params, a, b)
}

func fieldInvGPU(sess *accel.Session, params FieldParams, a []uint64) ([]uint64, error) {
	// Uses zk/field_inv kernel
	// Batch inversion via Montgomery's trick for efficiency
	// TODO: Implement kernel dispatch
	return fieldInvCPU(params, a)
}

func fieldExpGPU(sess *accel.Session, params FieldParams, a []uint64, exp uint64) ([]uint64, error) {
	// Uses zk/field_exp kernel with binary exponentiation
	// TODO: Implement kernel dispatch
	return fieldExpCPU(params, a, exp)
}
