//go:build accel

package consensus

import (
	"github.com/luxfi/accel"
)

// GPU-accelerated implementations using consensus kernels

func processVotesBatchGPU(sess *accel.Session, votes []VoteData) (int, error) {
	// Uses gpu/consensus kernel for parallel vote processing
	// TODO: Implement kernel dispatch via sess.Execute()
	return processVotesBatchCPU(votes)
}

func computeQuorumGPU(sess *accel.Session, votes []VoteData, validators []ValidatorWeight, threshold float64) (*QuorumResult, error) {
	// Uses gpu/consensus kernel for parallel quorum computation
	// Efficiently aggregates weights using parallel reduction
	// TODO: Implement kernel dispatch
	return computeQuorumCPU(votes, validators, threshold)
}

func aggregateVotesGPU(sess *accel.Session, votes []VoteData, validators []ValidatorWeight) (map[[32]byte]uint64, error) {
	// Uses gpu/consensus kernel for parallel vote aggregation
	// TODO: Implement kernel dispatch
	return aggregateVotesCPU(votes, validators)
}

func batchVerifyVoteSignaturesGPU(sess *accel.Session, votes []VoteData, signatures, pubkeys [][]byte) ([]bool, error) {
	// Uses crypto/ecdsa_batch or crypto/ed25519_batch kernel
	// TODO: Implement kernel dispatch
	return batchVerifyVoteSignaturesCPU(votes, signatures, pubkeys)
}

func computeValidatorSetHashGPU(sess *accel.Session, validators []ValidatorWeight) ([32]byte, error) {
	// Uses crypto/sha256_batch kernel for parallel hashing
	// TODO: Implement kernel dispatch
	return computeValidatorSetHashCPU(validators)
}
