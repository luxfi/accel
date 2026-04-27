// Package crypto provides GPU-accelerated cryptographic operations.
//
// All operations use the unified accel backend selection and fallback to CPU.
// Backend is selected at runtime via GPU_BACKEND env or auto-detection.
package crypto

import (
	"errors"

	"github.com/luxfi/accel"
)

// SigType identifies the signature algorithm.
type SigType uint8

const (
	SigECDSA   SigType = iota // secp256k1
	SigEd25519                // Ed25519
	SigBLS                    // BLS12-381
	SigMLDSA65                // Post-quantum ML-DSA-65
)

// HashType identifies the hash algorithm.
type HashType uint8

const (
	HashSHA256 HashType = iota
	HashKeccak256
	HashBlake3
	HashPoseidon
)

var (
	ErrInvalidInput = errors.New("crypto: invalid input")
	ErrBatchEmpty   = errors.New("crypto: empty batch")
	ErrVerifyFailed = errors.New("crypto: verification failed")
)

// BatchVerify verifies multiple signatures in parallel.
// Returns a slice of bools indicating which signatures are valid.
// Uses GPU acceleration when available, falls back to CPU.
func BatchVerify(sigType SigType, sigs, msgs, pubkeys [][]byte) ([]bool, error) {
	if len(sigs) == 0 || len(sigs) != len(msgs) || len(sigs) != len(pubkeys) {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return batchVerifyCPU(sigType, sigs, msgs, pubkeys)
	}
	defer sess.Close()

	return batchVerifyGPU(sess, sigType, sigs, msgs, pubkeys)
}

// Hash computes hashes for multiple inputs in parallel.
func Hash(hashType HashType, inputs [][]byte) ([][32]byte, error) {
	if len(inputs) == 0 {
		return nil, ErrBatchEmpty
	}

	sess, err := accel.NewSession()
	if err != nil {
		return hashCPU(hashType, inputs)
	}
	defer sess.Close()

	return hashGPU(sess, hashType, inputs)
}

// MSM performs multi-scalar multiplication on elliptic curves.
// Used for BLS signature aggregation and other EC operations.
func MSM(curve string, scalars, points [][]byte) ([]byte, error) {
	if len(scalars) != len(points) || len(scalars) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return msmCPU(curve, scalars, points)
	}
	defer sess.Close()

	return msmGPU(sess, curve, scalars, points)
}

// AggregateSignatures aggregates BLS signatures.
func AggregateSignatures(sigs [][]byte) ([]byte, error) {
	if len(sigs) == 0 {
		return nil, ErrBatchEmpty
	}

	sess, err := accel.NewSession()
	if err != nil {
		return aggregateSigsCPU(sigs)
	}
	defer sess.Close()

	return aggregateSigsGPU(sess, sigs)
}

// AggregatePublicKeys aggregates BLS public keys.
func AggregatePublicKeys(pks [][]byte) ([]byte, error) {
	if len(pks) == 0 {
		return nil, ErrBatchEmpty
	}

	sess, err := accel.NewSession()
	if err != nil {
		return aggregatePksCPU(pks)
	}
	defer sess.Close()

	return aggregatePksGPU(sess, pks)
}
