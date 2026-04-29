//go:build !accel

package crypto

import (
	"github.com/luxfi/accel"
)

// When accel build tag is not set, GPU functions fall back to CPU

func batchVerifyGPU(sess *accel.Session, sigType SigType, sigs, msgs, pubkeys [][]byte) ([]bool, error) {
	return batchVerifyCPU(sigType, sigs, msgs, pubkeys)
}

func hashGPU(sess *accel.Session, hashType HashType, inputs [][]byte) ([][32]byte, error) {
	return hashCPU(hashType, inputs)
}

func msmGPU(sess *accel.Session, curve Curve, scalars, points [][]byte) ([]byte, error) {
	return msmCPU(curve, scalars, points)
}

func aggregateSigsGPU(sess *accel.Session, signatures [][]byte) ([]byte, error) {
	return aggregateSigsCPU(signatures)
}

func aggregatePksGPU(sess *accel.Session, pubkeys [][]byte) ([]byte, error) {
	return aggregatePksCPU(pubkeys)
}
