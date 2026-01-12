//go:build !accel

package lattice

import (
	"github.com/luxfi/accel"
)

// When accel build tag is not set, GPU functions fall back to CPU

func nttForwardGPU(sess *accel.Session, params NTTParams, coeffs []uint64) ([]uint64, error) {
	return nttForwardCPU(params, coeffs)
}

func nttInverseGPU(sess *accel.Session, params NTTParams, evals []uint64) ([]uint64, error) {
	return nttInverseCPU(params, evals)
}

func polyMulGPU(sess *accel.Session, params NTTParams, a, b []uint64) ([]uint64, error) {
	return polyMulCPU(params, a, b)
}

func polyAddGPU(sess *accel.Session, modulus uint64, a, b []uint64) ([]uint64, error) {
	return polyAddCPU(modulus, a, b)
}

func polySubGPU(sess *accel.Session, modulus uint64, a, b []uint64) ([]uint64, error) {
	return polySubCPU(modulus, a, b)
}

func batchNTTForwardGPU(sess *accel.Session, params NTTParams, polys [][]uint64) ([][]uint64, error) {
	return batchNTTForwardCPU(params, polys)
}

func batchNTTInverseGPU(sess *accel.Session, params NTTParams, polys [][]uint64) ([][]uint64, error) {
	return batchNTTInverseCPU(params, polys)
}

func sampleNTTGPU(sess *accel.Session, params NTTParams, seed []byte) ([]uint64, error) {
	return sampleNTTCPU(params, seed)
}
