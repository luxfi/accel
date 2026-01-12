//go:build !accel

package zk

import (
	"github.com/luxfi/accel"
)

// When accel build tag is not set, GPU functions fall back to CPU implementations.

func nttGPU(sess *accel.Session, params NTTParams, coeffs []uint64) ([]uint64, error) {
	return nttCPU(params, coeffs)
}

func inttGPU(sess *accel.Session, params NTTParams, evals []uint64) ([]uint64, error) {
	return inttCPU(params, evals)
}

func batchNTTGPU(sess *accel.Session, params NTTParams, polys [][]uint64) ([][]uint64, error) {
	return batchNTTCPU(params, polys)
}

func batchINTTGPU(sess *accel.Session, params NTTParams, polys [][]uint64) ([][]uint64, error) {
	return batchINTTCPU(params, polys)
}

func msmGPU(sess *accel.Session, curve CurveType, scalars, points [][]byte) ([]byte, error) {
	return msmCPU(curve, scalars, points)
}

func msmBatchGPU(sess *accel.Session, curve CurveType, scalars, points [][][]byte) ([][]byte, error) {
	return msmBatchCPU(curve, scalars, points)
}

func poseidon2GPU(sess *accel.Session, params Poseidon2Params, inputs []uint64) (uint64, error) {
	return poseidon2CPU(params, inputs)
}

func poseidon2HashGPU(sess *accel.Session, params Poseidon2Params, left, right uint64) (uint64, error) {
	return poseidon2HashCPU(params, left, right)
}

func batchPoseidon2HashGPU(sess *accel.Session, params Poseidon2Params, lefts, rights []uint64) ([]uint64, error) {
	return batchPoseidon2HashCPU(params, lefts, rights)
}

func polyAddGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	return polyAddCPU(params, a, b)
}

func polySubGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	return polySubCPU(params, a, b)
}

func polyMulGPU(sess *accel.Session, params NTTParams, a, b []uint64) ([]uint64, error) {
	return polyMulCPU(params, a, b)
}

func polyMulPointwiseGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	return polyMulPointwiseCPU(params, a, b)
}

func polyEvalGPU(sess *accel.Session, params FieldParams, coeffs, points []uint64) ([]uint64, error) {
	return polyEvalCPU(params, coeffs, points)
}

func polyInterpolateGPU(sess *accel.Session, params FieldParams, xs, ys []uint64) ([]uint64, error) {
	return polyInterpolateCPU(params, xs, ys)
}

func friFoldGPU(sess *accel.Session, params FRIParams, evals []uint64, alpha uint64) ([]uint64, error) {
	return friFoldCPU(params, evals, alpha)
}

func friQueryPhaseGPU(sess *accel.Session, params FRIParams, evals []uint64, indices []uint32) ([][]uint64, error) {
	return friQueryPhaseCPU(params, evals, indices)
}

func commitPolyGPU(sess *accel.Session, params CommitParams, coeffs [][]byte, srs [][]byte) ([]byte, error) {
	return commitPolyCPU(params, coeffs, srs)
}

func batchCommitPolyGPU(sess *accel.Session, params CommitParams, coeffsList [][][]byte, srs [][]byte) ([][]byte, error) {
	return batchCommitPolyCPU(params, coeffsList, srs)
}

func fieldAddGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	return fieldAddCPU(params, a, b)
}

func fieldMulGPU(sess *accel.Session, params FieldParams, a, b []uint64) ([]uint64, error) {
	return fieldMulCPU(params, a, b)
}

func fieldInvGPU(sess *accel.Session, params FieldParams, a []uint64) ([]uint64, error) {
	return fieldInvCPU(params, a)
}

func fieldExpGPU(sess *accel.Session, params FieldParams, a []uint64, exp uint64) ([]uint64, error) {
	return fieldExpCPU(params, a, exp)
}
