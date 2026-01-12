// Package zk provides GPU-accelerated zero-knowledge proof operations.
//
// Supports NTT, MSM, polynomial arithmetic, Poseidon2 hashing, FRI folding,
// and commitment schemes. All operations use unified accel backend selection
// with automatic fallback to CPU implementations.
package zk

import (
	"errors"

	"github.com/luxfi/accel"
)

// Errors
var (
	ErrInvalidInput    = errors.New("zk: invalid input")
	ErrInvalidDegree   = errors.New("zk: invalid polynomial degree")
	ErrInvalidModulus  = errors.New("zk: invalid modulus")
	ErrDimensionMismatch = errors.New("zk: dimension mismatch")
	ErrEmptyBatch      = errors.New("zk: empty batch")
)

// CurveType identifies the elliptic curve for MSM operations.
type CurveType uint8

const (
	CurveBN254     CurveType = iota // BN254 (alt_bn128)
	CurveBLS12_381                  // BLS12-381
	CurveBLS12_377                  // BLS12-377
	CurvePallas                     // Pallas (Zcash/Halo2)
	CurveVesta                      // Vesta (Zcash/Halo2)
)

// NTTParams contains parameters for Number Theoretic Transform.
type NTTParams struct {
	N       uint32 // Polynomial degree (power of 2)
	Modulus uint64 // Prime modulus
	Root    uint64 // Primitive N-th root of unity mod Modulus
}

// FieldParams contains parameters for finite field operations.
type FieldParams struct {
	Modulus uint64 // Prime modulus
}

// Poseidon2Params contains parameters for Poseidon2 hash.
type Poseidon2Params struct {
	T          uint32   // State width (t)
	D          uint32   // S-box degree (typically 5 or 7)
	RoundsF    uint32   // Full rounds
	RoundsP    uint32   // Partial rounds
	Modulus    uint64   // Field modulus
	RoundConst []uint64 // Round constants
	MDS        []uint64 // MDS matrix (t x t)
}

// CommitParams contains parameters for polynomial commitments.
type CommitParams struct {
	Curve  CurveType // Curve for commitment
	Degree uint32    // Maximum polynomial degree
}

// FRIParams contains parameters for FRI folding.
type FRIParams struct {
	Modulus    uint64 // Field modulus
	FoldFactor uint32 // Folding factor (typically 2 or 4)
	BlowupFactor uint32 // Reed-Solomon blowup factor
}

// NTT performs forward Number Theoretic Transform.
// Transforms polynomial from coefficient to evaluation form.
func NTT(params NTTParams, coeffs []uint64) ([]uint64, error) {
	if len(coeffs) != int(params.N) {
		return nil, ErrInvalidDegree
	}
	if params.Modulus == 0 {
		return nil, ErrInvalidModulus
	}

	sess, err := accel.NewSession()
	if err != nil {
		return nttCPU(params, coeffs)
	}
	defer sess.Close()

	return nttGPU(sess, params, coeffs)
}

// INTT performs inverse Number Theoretic Transform.
// Transforms polynomial from evaluation to coefficient form.
func INTT(params NTTParams, evals []uint64) ([]uint64, error) {
	if len(evals) != int(params.N) {
		return nil, ErrInvalidDegree
	}
	if params.Modulus == 0 {
		return nil, ErrInvalidModulus
	}

	sess, err := accel.NewSession()
	if err != nil {
		return inttCPU(params, evals)
	}
	defer sess.Close()

	return inttGPU(sess, params, evals)
}

// BatchNTT performs forward NTT on multiple polynomials in parallel.
func BatchNTT(params NTTParams, polys [][]uint64) ([][]uint64, error) {
	if len(polys) == 0 {
		return nil, ErrEmptyBatch
	}
	for _, p := range polys {
		if len(p) != int(params.N) {
			return nil, ErrInvalidDegree
		}
	}

	sess, err := accel.NewSession()
	if err != nil {
		return batchNTTCPU(params, polys)
	}
	defer sess.Close()

	return batchNTTGPU(sess, params, polys)
}

// BatchINTT performs inverse NTT on multiple polynomials in parallel.
func BatchINTT(params NTTParams, polys [][]uint64) ([][]uint64, error) {
	if len(polys) == 0 {
		return nil, ErrEmptyBatch
	}
	for _, p := range polys {
		if len(p) != int(params.N) {
			return nil, ErrInvalidDegree
		}
	}

	sess, err := accel.NewSession()
	if err != nil {
		return batchINTTCPU(params, polys)
	}
	defer sess.Close()

	return batchINTTGPU(sess, params, polys)
}

// MSM performs multi-scalar multiplication.
// result = sum(scalars[i] * points[i]) for all i
func MSM(curve CurveType, scalars, points [][]byte) ([]byte, error) {
	if len(scalars) != len(points) {
		return nil, ErrDimensionMismatch
	}
	if len(scalars) == 0 {
		return nil, ErrEmptyBatch
	}

	sess, err := accel.NewSession()
	if err != nil {
		return msmCPU(curve, scalars, points)
	}
	defer sess.Close()

	return msmGPU(sess, curve, scalars, points)
}

// MSMBatch performs multiple MSMs in parallel.
func MSMBatch(curve CurveType, scalars, points [][][]byte) ([][]byte, error) {
	if len(scalars) != len(points) {
		return nil, ErrDimensionMismatch
	}
	if len(scalars) == 0 {
		return nil, ErrEmptyBatch
	}

	sess, err := accel.NewSession()
	if err != nil {
		return msmBatchCPU(curve, scalars, points)
	}
	defer sess.Close()

	return msmBatchGPU(sess, curve, scalars, points)
}

// Poseidon2 computes Poseidon2 hash of inputs.
// Returns field element as uint64 (for fields <= 64 bits) or []byte.
func Poseidon2(params Poseidon2Params, inputs []uint64) (uint64, error) {
	if len(inputs) == 0 {
		return 0, ErrInvalidInput
	}
	if params.Modulus == 0 {
		return 0, ErrInvalidModulus
	}

	sess, err := accel.NewSession()
	if err != nil {
		return poseidon2CPU(params, inputs)
	}
	defer sess.Close()

	return poseidon2GPU(sess, params, inputs)
}

// Poseidon2Hash computes Poseidon2 hash for Merkle trees.
// left and right are field elements to hash together.
func Poseidon2Hash(params Poseidon2Params, left, right uint64) (uint64, error) {
	if params.Modulus == 0 {
		return 0, ErrInvalidModulus
	}

	sess, err := accel.NewSession()
	if err != nil {
		return poseidon2HashCPU(params, left, right)
	}
	defer sess.Close()

	return poseidon2HashGPU(sess, params, left, right)
}

// BatchPoseidon2Hash computes multiple Poseidon2 hashes in parallel.
// Used for building Merkle trees efficiently.
func BatchPoseidon2Hash(params Poseidon2Params, lefts, rights []uint64) ([]uint64, error) {
	if len(lefts) != len(rights) {
		return nil, ErrDimensionMismatch
	}
	if len(lefts) == 0 {
		return nil, ErrEmptyBatch
	}

	sess, err := accel.NewSession()
	if err != nil {
		return batchPoseidon2HashCPU(params, lefts, rights)
	}
	defer sess.Close()

	return batchPoseidon2HashGPU(sess, params, lefts, rights)
}

// PolyAdd adds two polynomials coefficient-wise.
func PolyAdd(params FieldParams, a, b []uint64) ([]uint64, error) {
	if len(a) != len(b) {
		return nil, ErrDimensionMismatch
	}
	if len(a) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return polyAddCPU(params, a, b)
	}
	defer sess.Close()

	return polyAddGPU(sess, params, a, b)
}

// PolySub subtracts two polynomials coefficient-wise.
func PolySub(params FieldParams, a, b []uint64) ([]uint64, error) {
	if len(a) != len(b) {
		return nil, ErrDimensionMismatch
	}
	if len(a) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return polySubCPU(params, a, b)
	}
	defer sess.Close()

	return polySubGPU(sess, params, a, b)
}

// PolyMul multiplies two polynomials using NTT.
// Result is in coefficient form with degree len(a) + len(b) - 1.
func PolyMul(params NTTParams, a, b []uint64) ([]uint64, error) {
	if len(a) != int(params.N) || len(b) != int(params.N) {
		return nil, ErrInvalidDegree
	}

	sess, err := accel.NewSession()
	if err != nil {
		return polyMulCPU(params, a, b)
	}
	defer sess.Close()

	return polyMulGPU(sess, params, a, b)
}

// PolyMulPointwise multiplies polynomials in NTT domain (pointwise).
func PolyMulPointwise(params FieldParams, a, b []uint64) ([]uint64, error) {
	if len(a) != len(b) {
		return nil, ErrDimensionMismatch
	}

	sess, err := accel.NewSession()
	if err != nil {
		return polyMulPointwiseCPU(params, a, b)
	}
	defer sess.Close()

	return polyMulPointwiseGPU(sess, params, a, b)
}

// PolyEval evaluates polynomial at given points.
// coeffs: [degree+1] coefficients, points: evaluation points
// Returns values[i] = poly(points[i])
func PolyEval(params FieldParams, coeffs, points []uint64) ([]uint64, error) {
	if len(coeffs) == 0 || len(points) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return polyEvalCPU(params, coeffs, points)
	}
	defer sess.Close()

	return polyEvalGPU(sess, params, coeffs, points)
}

// PolyInterpolate computes polynomial from points using Lagrange interpolation.
// xs, ys: evaluation points and values
// Returns polynomial coefficients.
func PolyInterpolate(params FieldParams, xs, ys []uint64) ([]uint64, error) {
	if len(xs) != len(ys) {
		return nil, ErrDimensionMismatch
	}
	if len(xs) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return polyInterpolateCPU(params, xs, ys)
	}
	defer sess.Close()

	return polyInterpolateGPU(sess, params, xs, ys)
}

// FRIFold performs FRI folding step.
// evals: polynomial evaluations at rate-2 LDE
// alpha: random folding challenge
// Returns folded polynomial evaluations.
func FRIFold(params FRIParams, evals []uint64, alpha uint64) ([]uint64, error) {
	if len(evals) == 0 || len(evals)&1 != 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return friFoldCPU(params, evals, alpha)
	}
	defer sess.Close()

	return friFoldGPU(sess, params, evals, alpha)
}

// FRIQueryPhase computes query responses for FRI protocol.
// evals: committed polynomial evaluations
// indices: query indices
// Returns query response data.
func FRIQueryPhase(params FRIParams, evals []uint64, indices []uint32) ([][]uint64, error) {
	if len(evals) == 0 || len(indices) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return friQueryPhaseCPU(params, evals, indices)
	}
	defer sess.Close()

	return friQueryPhaseGPU(sess, params, evals, indices)
}

// CommitPoly computes a polynomial commitment (KZG).
// coeffs: polynomial coefficients
// srs: structured reference string (powers of tau)
// Returns commitment point in compressed form.
func CommitPoly(params CommitParams, coeffs [][]byte, srs [][]byte) ([]byte, error) {
	if len(coeffs) == 0 {
		return nil, ErrInvalidInput
	}
	if uint32(len(coeffs)) > params.Degree+1 {
		return nil, ErrInvalidDegree
	}

	sess, err := accel.NewSession()
	if err != nil {
		return commitPolyCPU(params, coeffs, srs)
	}
	defer sess.Close()

	return commitPolyGPU(sess, params, coeffs, srs)
}

// BatchCommitPoly computes multiple polynomial commitments in parallel.
func BatchCommitPoly(params CommitParams, coeffsList [][][]byte, srs [][]byte) ([][]byte, error) {
	if len(coeffsList) == 0 {
		return nil, ErrEmptyBatch
	}

	sess, err := accel.NewSession()
	if err != nil {
		return batchCommitPolyCPU(params, coeffsList, srs)
	}
	defer sess.Close()

	return batchCommitPolyGPU(sess, params, coeffsList, srs)
}

// FieldAdd adds field elements.
func FieldAdd(params FieldParams, a, b []uint64) ([]uint64, error) {
	if len(a) != len(b) {
		return nil, ErrDimensionMismatch
	}

	sess, err := accel.NewSession()
	if err != nil {
		return fieldAddCPU(params, a, b)
	}
	defer sess.Close()

	return fieldAddGPU(sess, params, a, b)
}

// FieldMul multiplies field elements.
func FieldMul(params FieldParams, a, b []uint64) ([]uint64, error) {
	if len(a) != len(b) {
		return nil, ErrDimensionMismatch
	}

	sess, err := accel.NewSession()
	if err != nil {
		return fieldMulCPU(params, a, b)
	}
	defer sess.Close()

	return fieldMulGPU(sess, params, a, b)
}

// FieldInv computes modular inverse of field elements.
func FieldInv(params FieldParams, a []uint64) ([]uint64, error) {
	if len(a) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return fieldInvCPU(params, a)
	}
	defer sess.Close()

	return fieldInvGPU(sess, params, a)
}

// FieldExp computes a^exp mod modulus for each element.
func FieldExp(params FieldParams, a []uint64, exp uint64) ([]uint64, error) {
	if len(a) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return fieldExpCPU(params, a, exp)
	}
	defer sess.Close()

	return fieldExpGPU(sess, params, a, exp)
}
