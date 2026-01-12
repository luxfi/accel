//go:build accel

package crypto

import (
	"github.com/luxfi/accel"
)

// GPU-accelerated implementations

func batchVerifyGPU(sess *accel.Session, sigType SigType, sigs, msgs, pubkeys [][]byte) ([]bool, error) {
	// Map signature type to kernel name
	var kernelName string
	switch sigType {
	case SigECDSA:
		kernelName = "crypto/secp256k1"
	case SigEd25519:
		kernelName = "crypto/ed25519"
	case SigBLS:
		kernelName = "crypto/bls12_381"
	case SigMLDSA65:
		kernelName = "crypto/mldsa_verify"
	}

	// Execute batch verification kernel
	results := make([]bool, len(sigs))

	// TODO: Implement actual kernel dispatch via Session.Execute()
	// For now, fallback to CPU
	return batchVerifyCPU(sigType, sigs, msgs, pubkeys)

	_ = kernelName
	return results, nil
}

func hashGPU(sess *accel.Session, hashType HashType, inputs [][]byte) ([][32]byte, error) {
	var kernelName string
	switch hashType {
	case HashSHA256:
		kernelName = "crypto/sha256"
	case HashKeccak256:
		kernelName = "crypto/keccak256"
	case HashBlake3:
		kernelName = "crypto/blake3"
	case HashPoseidon:
		kernelName = "crypto/poseidon"
	}

	// TODO: Implement kernel dispatch
	_ = kernelName
	return hashCPU(hashType, inputs)
}

func msmGPU(sess *accel.Session, curve string, scalars, points [][]byte) ([]byte, error) {
	// Use crypto/msm kernel
	// TODO: Implement kernel dispatch
	return msmCPU(curve, scalars, points)
}

func aggregateSigsGPU(sess *accel.Session, sigs [][]byte) ([]byte, error) {
	// Use crypto/bls12_381 aggregation
	return aggregateSigsCPU(sigs)
}

func aggregatePksGPU(sess *accel.Session, pks [][]byte) ([]byte, error) {
	return aggregatePksCPU(pks)
}
