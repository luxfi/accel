// Package consensus provides GPU-accelerated consensus primitives.
//
// Used for accelerating vote processing, validator set computations,
// and other consensus-critical operations that can benefit from parallelization.
package consensus

import (
	"errors"

	"github.com/luxfi/accel"
)

var (
	ErrInvalidInput     = errors.New("consensus: invalid input")
	ErrInvalidBatchSize = errors.New("consensus: invalid batch size")
	ErrNotInitialized   = errors.New("consensus: not initialized")
)

// VoteData represents a vote for batch processing.
type VoteData struct {
	VoterID      [32]byte
	BlockID      [32]byte
	IsPreference bool
}

// ValidatorWeight represents a validator and its stake weight.
type ValidatorWeight struct {
	ValidatorID [32]byte
	Weight      uint64
}

// QuorumResult contains the result of a quorum check.
type QuorumResult struct {
	HasQuorum    bool
	TotalWeight  uint64
	VotedWeight  uint64
	QuorumWeight uint64
}

// ProcessVotesBatch processes a batch of votes in parallel.
// Returns the number of votes successfully processed.
func ProcessVotesBatch(votes []VoteData) (int, error) {
	if len(votes) == 0 {
		return 0, nil
	}

	sess, err := accel.NewSession()
	if err != nil {
		return processVotesBatchCPU(votes)
	}
	defer sess.Close()

	return processVotesBatchGPU(sess, votes)
}

// ComputeQuorum checks if a quorum is reached given votes and validators.
// threshold is the fraction of stake required (e.g., 0.67 for 2/3).
func ComputeQuorum(votes []VoteData, validators []ValidatorWeight, threshold float64) (*QuorumResult, error) {
	if len(votes) == 0 || len(validators) == 0 {
		return nil, ErrInvalidInput
	}
	if threshold <= 0 || threshold > 1 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return computeQuorumCPU(votes, validators, threshold)
	}
	defer sess.Close()

	return computeQuorumGPU(sess, votes, validators, threshold)
}

// AggregateVotes aggregates votes by block, returning vote counts per block.
// Returns a map of block ID to total weight voted for that block.
func AggregateVotes(votes []VoteData, validators []ValidatorWeight) (map[[32]byte]uint64, error) {
	if len(votes) == 0 {
		return make(map[[32]byte]uint64), nil
	}

	sess, err := accel.NewSession()
	if err != nil {
		return aggregateVotesCPU(votes, validators)
	}
	defer sess.Close()

	return aggregateVotesGPU(sess, votes, validators)
}

// BatchVerifyVoteSignatures verifies vote signatures in parallel.
// Returns a slice of booleans indicating validity of each vote.
func BatchVerifyVoteSignatures(votes []VoteData, signatures, pubkeys [][]byte) ([]bool, error) {
	if len(votes) != len(signatures) || len(votes) != len(pubkeys) {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return batchVerifyVoteSignaturesCPU(votes, signatures, pubkeys)
	}
	defer sess.Close()

	return batchVerifyVoteSignaturesGPU(sess, votes, signatures, pubkeys)
}

// ComputeValidatorSetHash computes a hash of the validator set for checkpointing.
func ComputeValidatorSetHash(validators []ValidatorWeight) ([32]byte, error) {
	if len(validators) == 0 {
		return [32]byte{}, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return computeValidatorSetHashCPU(validators)
	}
	defer sess.Close()

	return computeValidatorSetHashGPU(sess, validators)
}
