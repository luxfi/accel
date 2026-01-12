package crypto

import (
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/sha256"

	"golang.org/x/crypto/blake2b"
	"golang.org/x/crypto/sha3"
)

// CPU fallback implementations

func batchVerifyCPU(sigType SigType, sigs, msgs, pubkeys [][]byte) ([]bool, error) {
	results := make([]bool, len(sigs))

	for i := range sigs {
		switch sigType {
		case SigEd25519:
			if len(pubkeys[i]) == ed25519.PublicKeySize && len(sigs[i]) == ed25519.SignatureSize {
				results[i] = ed25519.Verify(pubkeys[i], msgs[i], sigs[i])
			}
		case SigECDSA:
			// Simplified - real impl would parse DER signature
			results[i] = verifyECDSA(pubkeys[i], msgs[i], sigs[i])
		case SigBLS:
			// BLS verification - delegate to crypto library
			results[i] = verifyBLS(pubkeys[i], msgs[i], sigs[i])
		case SigMLDSA65:
			// ML-DSA verification
			results[i] = verifyMLDSA(pubkeys[i], msgs[i], sigs[i])
		}
	}

	return results, nil
}

func hashCPU(hashType HashType, inputs [][]byte) ([][32]byte, error) {
	results := make([][32]byte, len(inputs))

	for i, input := range inputs {
		switch hashType {
		case HashSHA256:
			results[i] = sha256.Sum256(input)
		case HashKeccak256:
			h := sha3.NewLegacyKeccak256()
			h.Write(input)
			copy(results[i][:], h.Sum(nil))
		case HashBlake3:
			h, _ := blake2b.New256(nil)
			h.Write(input)
			copy(results[i][:], h.Sum(nil))
		case HashPoseidon:
			// Poseidon hash - would use specialized library
			results[i] = poseidonHash(input)
		}
	}

	return results, nil
}

func msmCPU(curve string, scalars, points [][]byte) ([]byte, error) {
	// CPU MSM implementation - would use gnark-crypto or similar
	// For now, placeholder
	return nil, ErrVerifyFailed
}

func aggregateSigsCPU(sigs [][]byte) ([]byte, error) {
	// BLS signature aggregation - would use bls12-381 library
	return nil, nil
}

func aggregatePksCPU(pks [][]byte) ([]byte, error) {
	// BLS public key aggregation
	return nil, nil
}

// Stub implementations - would be replaced with real crypto libs
func verifyECDSA(pk, msg, sig []byte) bool {
	// Parse and verify ECDSA secp256k1
	_ = ecdsa.PublicKey{}
	return false
}

func verifyBLS(pk, msg, sig []byte) bool {
	// BLS12-381 verification
	return false
}

func verifyMLDSA(pk, msg, sig []byte) bool {
	// ML-DSA post-quantum verification
	return false
}

func poseidonHash(input []byte) [32]byte {
	// Poseidon hash for ZK-friendly hashing
	return sha256.Sum256(input) // Placeholder
}
