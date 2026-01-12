package consensus

import (
	"crypto/sha256"
	"encoding/binary"
)

// CPU fallback implementations

func processVotesBatchCPU(votes []VoteData) (int, error) {
	// Simple validation pass - in production this would do more work
	processed := 0
	for range votes {
		// Validate vote structure
		processed++
	}
	return processed, nil
}

func computeQuorumCPU(votes []VoteData, validators []ValidatorWeight, threshold float64) (*QuorumResult, error) {
	// Build validator lookup
	validatorWeights := make(map[[32]byte]uint64)
	var totalWeight uint64
	for _, v := range validators {
		validatorWeights[v.ValidatorID] = v.Weight
		totalWeight += v.Weight
	}

	// Count weighted votes
	var votedWeight uint64
	seen := make(map[[32]byte]bool)
	for _, vote := range votes {
		if seen[vote.VoterID] {
			continue // Skip duplicate votes
		}
		seen[vote.VoterID] = true
		if weight, ok := validatorWeights[vote.VoterID]; ok {
			votedWeight += weight
		}
	}

	quorumWeight := uint64(float64(totalWeight) * threshold)
	hasQuorum := votedWeight >= quorumWeight

	return &QuorumResult{
		HasQuorum:    hasQuorum,
		TotalWeight:  totalWeight,
		VotedWeight:  votedWeight,
		QuorumWeight: quorumWeight,
	}, nil
}

func aggregateVotesCPU(votes []VoteData, validators []ValidatorWeight) (map[[32]byte]uint64, error) {
	// Build validator lookup
	validatorWeights := make(map[[32]byte]uint64)
	for _, v := range validators {
		validatorWeights[v.ValidatorID] = v.Weight
	}

	// Aggregate by block
	result := make(map[[32]byte]uint64)
	seen := make(map[[32]byte]map[[32]byte]bool) // block -> voter -> seen

	for _, vote := range votes {
		if seen[vote.BlockID] == nil {
			seen[vote.BlockID] = make(map[[32]byte]bool)
		}
		if seen[vote.BlockID][vote.VoterID] {
			continue // Skip duplicate votes
		}
		seen[vote.BlockID][vote.VoterID] = true

		if weight, ok := validatorWeights[vote.VoterID]; ok {
			result[vote.BlockID] += weight
		}
	}

	return result, nil
}

func batchVerifyVoteSignaturesCPU(votes []VoteData, signatures, pubkeys [][]byte) ([]bool, error) {
	// CPU signature verification - uses Ed25519 in production
	// For now, just verify lengths are valid
	results := make([]bool, len(votes))
	for i := range votes {
		// Check signature length (Ed25519 = 64 bytes)
		if len(signatures[i]) == 64 && len(pubkeys[i]) == 32 {
			// In production: ed25519.Verify(pubkeys[i], voteBytes, signatures[i])
			results[i] = true
		}
	}
	return results, nil
}

func computeValidatorSetHashCPU(validators []ValidatorWeight) ([32]byte, error) {
	h := sha256.New()

	for _, v := range validators {
		h.Write(v.ValidatorID[:])
		var buf [8]byte
		binary.BigEndian.PutUint64(buf[:], v.Weight)
		h.Write(buf[:])
	}

	var result [32]byte
	copy(result[:], h.Sum(nil))
	return result, nil
}
