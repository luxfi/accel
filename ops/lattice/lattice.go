// Package lattice provides GPU-accelerated lattice cryptography operations.
//
// Used for FHE (Fully Homomorphic Encryption), post-quantum signatures,
// and other lattice-based cryptographic schemes.
package lattice

import (
	"errors"

	"github.com/luxfi/accel"
)

var (
	ErrInvalidInput  = errors.New("lattice: invalid input")
	ErrInvalidDegree = errors.New("lattice: invalid polynomial degree")
	ErrInvalidModulus = errors.New("lattice: invalid modulus")
)

// NTTParams contains parameters for NTT operations.
type NTTParams struct {
	N       uint32 // Polynomial degree (power of 2)
	Modulus uint64 // Prime modulus
	Root    uint64 // Primitive N-th root of unity
}

// NTTForward performs forward Number Theoretic Transform.
// Transforms polynomial from coefficient to evaluation form.
func NTTForward(params NTTParams, coeffs []uint64) ([]uint64, error) {
	if len(coeffs) != int(params.N) {
		return nil, ErrInvalidDegree
	}

	sess, err := accel.NewSession()
	if err != nil {
		return nttForwardCPU(params, coeffs)
	}
	defer sess.Close()

	return nttForwardGPU(sess, params, coeffs)
}

// NTTInverse performs inverse Number Theoretic Transform.
// Transforms polynomial from evaluation to coefficient form.
func NTTInverse(params NTTParams, evals []uint64) ([]uint64, error) {
	if len(evals) != int(params.N) {
		return nil, ErrInvalidDegree
	}

	sess, err := accel.NewSession()
	if err != nil {
		return nttInverseCPU(params, evals)
	}
	defer sess.Close()

	return nttInverseGPU(sess, params, evals)
}

// PolyMul multiplies two polynomials using NTT.
// Result is (a * b) mod (X^N + 1) mod modulus.
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

// PolyAdd adds two polynomials coefficient-wise.
func PolyAdd(modulus uint64, a, b []uint64) ([]uint64, error) {
	if len(a) != len(b) {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return polyAddCPU(modulus, a, b)
	}
	defer sess.Close()

	return polyAddGPU(sess, modulus, a, b)
}

// PolySub subtracts two polynomials coefficient-wise.
func PolySub(modulus uint64, a, b []uint64) ([]uint64, error) {
	if len(a) != len(b) {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return polySubCPU(modulus, a, b)
	}
	defer sess.Close()

	return polySubGPU(sess, modulus, a, b)
}

// BatchNTTForward performs forward NTT on multiple polynomials in parallel.
func BatchNTTForward(params NTTParams, polys [][]uint64) ([][]uint64, error) {
	for _, p := range polys {
		if len(p) != int(params.N) {
			return nil, ErrInvalidDegree
		}
	}

	sess, err := accel.NewSession()
	if err != nil {
		return batchNTTForwardCPU(params, polys)
	}
	defer sess.Close()

	return batchNTTForwardGPU(sess, params, polys)
}

// BatchNTTInverse performs inverse NTT on multiple polynomials in parallel.
func BatchNTTInverse(params NTTParams, polys [][]uint64) ([][]uint64, error) {
	for _, p := range polys {
		if len(p) != int(params.N) {
			return nil, ErrInvalidDegree
		}
	}

	sess, err := accel.NewSession()
	if err != nil {
		return batchNTTInverseCPU(params, polys)
	}
	defer sess.Close()

	return batchNTTInverseGPU(sess, params, polys)
}

// SampleNTT samples a polynomial in NTT domain from a seed.
// Used for generating random polynomials in lattice schemes.
func SampleNTT(params NTTParams, seed []byte) ([]uint64, error) {
	if len(seed) < 32 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return sampleNTTCPU(params, seed)
	}
	defer sess.Close()

	return sampleNTTGPU(sess, params, seed)
}
