//go:build !accel

package consensus

import (
	"github.com/luxfi/accel"
)

// When accel build tag is not set, GPU functions fall back to CPU

func processVotesBatchGPU(sess *accel.Session, votes []VoteData) (int, error) {
	return processVotesBatchCPU(votes)
}

func computeQuorumGPU(sess *accel.Session, votes []VoteData, validators []ValidatorWeight, threshold float64) (*QuorumResult, error) {
	return computeQuorumCPU(votes, validators, threshold)
}

func aggregateVotesGPU(sess *accel.Session, votes []VoteData, validators []ValidatorWeight) (map[[32]byte]uint64, error) {
	return aggregateVotesCPU(votes, validators)
}

func batchVerifyVoteSignaturesGPU(sess *accel.Session, votes []VoteData, signatures, pubkeys [][]byte) ([]bool, error) {
	return batchVerifyVoteSignaturesCPU(votes, signatures, pubkeys)
}

func computeValidatorSetHashGPU(sess *accel.Session, validators []ValidatorWeight) ([32]byte, error) {
	return computeValidatorSetHashCPU(validators)
}
