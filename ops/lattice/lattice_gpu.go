//go:build accel

package lattice

import (
	"github.com/luxfi/accel"
)

// GPU-accelerated implementations using lattice kernels

func nttForwardGPU(sess *accel.Session, params NTTParams, coeffs []uint64) ([]uint64, error) {
	// Uses lattice/ntt kernel for GPU-accelerated NTT
	// TODO: Implement kernel dispatch
	return nttForwardCPU(params, coeffs)
}

func nttInverseGPU(sess *accel.Session, params NTTParams, evals []uint64) ([]uint64, error) {
	// Uses lattice/ntt kernel for inverse NTT
	// TODO: Implement kernel dispatch
	return nttInverseCPU(params, evals)
}

func polyMulGPU(sess *accel.Session, params NTTParams, a, b []uint64) ([]uint64, error) {
	// Uses lattice/poly_mul kernel
	// TODO: Implement kernel dispatch
	return polyMulCPU(params, a, b)
}

func polyAddGPU(sess *accel.Session, modulus uint64, a, b []uint64) ([]uint64, error) {
	// Uses lattice/poly_arithmetic kernel
	// TODO: Implement kernel dispatch
	return polyAddCPU(modulus, a, b)
}

func polySubGPU(sess *accel.Session, modulus uint64, a, b []uint64) ([]uint64, error) {
	// Uses lattice/poly_arithmetic kernel
	return polySubCPU(modulus, a, b)
}

func batchNTTForwardGPU(sess *accel.Session, params NTTParams, polys [][]uint64) ([][]uint64, error) {
	// Uses fhe/ntt_four_step kernel for large batches
	// TODO: Implement kernel dispatch
	return batchNTTForwardCPU(params, polys)
}

func batchNTTInverseGPU(sess *accel.Session, params NTTParams, polys [][]uint64) ([][]uint64, error) {
	// Uses fhe/ntt_four_step kernel for large batches
	return batchNTTInverseCPU(params, polys)
}

func sampleNTTGPU(sess *accel.Session, params NTTParams, seed []byte) ([]uint64, error) {
	// Uses lattice/sample_ntt kernel
	return sampleNTTCPU(params, seed)
}
