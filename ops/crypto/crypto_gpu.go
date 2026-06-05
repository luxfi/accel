//go:build accel

package crypto

import (
	"errors"

	"github.com/luxfi/accel"
)

// GPU-accelerated cryptographic implementations.
// These dispatch to the accel library's GPU backends (CUDA, Metal, Vulkan)
// for parallel signature verification, hashing, and EC operations.

func batchVerifyGPU(sess *accel.Session, sigType SigType, sigs, msgs, pubkeys [][]byte) ([]bool, error) {
	n := len(sigs)
	if n == 0 {
		return []bool{}, nil
	}

	// Determine sizes based on signature type
	var sigSize, msgSize, pkSize int
	switch sigType {
	case SigECDSA:
		sigSize = 64 // r || s
		msgSize = 32 // message hash
		pkSize = 33  // compressed secp256k1 pubkey
	case SigEd25519:
		sigSize = 64 // Ed25519 signature
		msgSize = 32 // message (or hash)
		pkSize = 32  // Ed25519 public key
	case SigBLS:
		sigSize = 96 // BLS12-381 G2 point
		msgSize = 32 // message hash
		pkSize = 48  // BLS12-381 G1 point
	case SigMLDSA65:
		sigSize = 3309 // ML-DSA-65 signature
		msgSize = 32   // message hash
		pkSize = 1952  // ML-DSA-65 public key
	default:
		return batchVerifyCPU(sigType, sigs, msgs, pubkeys)
	}

	// Pack signatures into flat byte array
	sigBytes := make([]byte, n*sigSize)
	for i, sig := range sigs {
		if len(sig) >= sigSize {
			copy(sigBytes[i*sigSize:(i+1)*sigSize], sig[:sigSize])
		} else {
			copy(sigBytes[i*sigSize:i*sigSize+len(sig)], sig)
		}
	}

	// Pack messages
	msgBytes := make([]byte, n*msgSize)
	for i, msg := range msgs {
		if len(msg) >= msgSize {
			copy(msgBytes[i*msgSize:(i+1)*msgSize], msg[:msgSize])
		} else {
			copy(msgBytes[i*msgSize:i*msgSize+len(msg)], msg)
		}
	}

	// Pack public keys
	pkBytes := make([]byte, n*pkSize)
	for i, pk := range pubkeys {
		if len(pk) >= pkSize {
			copy(pkBytes[i*pkSize:(i+1)*pkSize], pk[:pkSize])
		} else {
			copy(pkBytes[i*pkSize:i*pkSize+len(pk)], pk)
		}
	}

	// Create GPU tensors
	msgTensor, err := accel.NewTensorWithData[uint8](sess, []int{n, msgSize}, msgBytes)
	if err != nil {
		return batchVerifyCPU(sigType, sigs, msgs, pubkeys)
	}
	defer msgTensor.Close()

	sigTensor, err := accel.NewTensorWithData[uint8](sess, []int{n, sigSize}, sigBytes)
	if err != nil {
		return batchVerifyCPU(sigType, sigs, msgs, pubkeys)
	}
	defer sigTensor.Close()

	pkTensor, err := accel.NewTensorWithData[uint8](sess, []int{n, pkSize}, pkBytes)
	if err != nil {
		return batchVerifyCPU(sigType, sigs, msgs, pubkeys)
	}
	defer pkTensor.Close()

	// Result tensor (1 byte per signature: 0 = invalid, 1 = valid)
	resultTensor, err := accel.NewTensor[uint8](sess, []int{n})
	if err != nil {
		return batchVerifyCPU(sigType, sigs, msgs, pubkeys)
	}
	defer resultTensor.Close()

	// Dispatch to appropriate GPU kernel
	cryptoOps := sess.Crypto()
	switch sigType {
	case SigECDSA:
		err = cryptoOps.ECDSAVerifyBatch(
			msgTensor.Untyped(),
			sigTensor.Untyped(),
			pkTensor.Untyped(),
			resultTensor.Untyped(),
		)
	case SigEd25519:
		err = cryptoOps.Ed25519VerifyBatch(
			msgTensor.Untyped(),
			sigTensor.Untyped(),
			pkTensor.Untyped(),
			resultTensor.Untyped(),
		)
	case SigBLS:
		err = cryptoOps.BLSVerifyBatch(
			msgTensor.Untyped(),
			sigTensor.Untyped(),
			pkTensor.Untyped(),
			resultTensor.Untyped(),
		)
	case SigMLDSA65:
		// Red CRITICAL #177: real per-element batch dispatch via the
		// FIPS 204 batch entry LatticeOps.MLDSAVerifyBatch. The prior
		// code called the single-sig DilithiumVerify ONCE for the
		// whole batch and broadcasted the single boolean to every
		// output position — pure consensus-split fuel. A batch of
		// [valid, INVALID, valid] returned [true, true, true] (or
		// [false, false, false] if the single dispatch happened to
		// short-circuit on the malformed entry), shipping wrong
		// verdicts into the EVM precompile output (mldsa/contract.go:
		// 339) for any GPU-accelerated validator. CPU-only validators
		// (batchVerifyCPU computes per-element verdicts) would see
		// [true, false, true]. Verdict asymmetry across the validator
		// set = consensus split on the EVM precompile path.
		//
		// MLDSAVerifyBatch writes per-element verdicts into the
		// shared resultTensor[n] uint8 — same shape as the
		// ECDSA/Ed25519/BLS cases above, so the unified Sync + ToSlice
		// + bytes→bool conversion below handles MLDSA identically.
		//
		// Honor the Red M-1 / CRITICAL #176 propagation policy:
		// ErrInvalidArgument from the C ABI is a HARD error —
		// propagate it so the precompile fails closed. Silent CPU
		// fallback here would let CPU-only validators accept input
		// the GPU plugin already declared malformed → asymmetric
		// verdicts → consensus split. Other accel errors
		// (NotSupported, OutOfMemory, KernelFailed) are recoverable;
		// fall back to CPU per the existing pattern.
		err = sess.Lattice().MLDSAVerifyBatch(
			accel.MLDSAMode65,
			msgTensor.Untyped(),
			sigTensor.Untyped(),
			pkTensor.Untyped(),
			resultTensor.Untyped(),
		)
		if err != nil && errors.Is(err, accel.ErrInvalidArgument) {
			return nil, err
		}
	}

	if err != nil {
		return batchVerifyCPU(sigType, sigs, msgs, pubkeys)
	}

	// Sync GPU operations
	if err := sess.Sync(); err != nil {
		return batchVerifyCPU(sigType, sigs, msgs, pubkeys)
	}

	// Read results
	resultBytes, err := resultTensor.ToSlice()
	if err != nil {
		return batchVerifyCPU(sigType, sigs, msgs, pubkeys)
	}

	results := make([]bool, n)
	for i, r := range resultBytes {
		results[i] = r != 0
	}

	return results, nil
}

func hashGPU(sess *accel.Session, hashType HashType, inputs [][]byte) ([][32]byte, error) {
	n := len(inputs)
	if n == 0 {
		return [][32]byte{}, nil
	}

	// Find max input length for padding
	maxLen := 0
	for _, input := range inputs {
		if len(input) > maxLen {
			maxLen = len(input)
		}
	}

	// Pad all inputs to same length (required for batched GPU processing)
	// Use zero-padding which is safe for these hash functions
	inputBytes := make([]byte, n*maxLen)
	for i, input := range inputs {
		copy(inputBytes[i*maxLen:i*maxLen+len(input)], input)
	}

	// Create input tensor
	inputTensor, err := accel.NewTensorWithData[uint8](sess, []int{n, maxLen}, inputBytes)
	if err != nil {
		return hashCPU(hashType, inputs)
	}
	defer inputTensor.Close()

	// Create output tensor (32 bytes per hash)
	outputTensor, err := accel.NewTensor[uint8](sess, []int{n, 32})
	if err != nil {
		return hashCPU(hashType, inputs)
	}
	defer outputTensor.Close()

	// Dispatch to appropriate hash kernel
	cryptoOps := sess.Crypto()
	switch hashType {
	case HashSHA256:
		err = cryptoOps.SHA256(inputTensor.Untyped(), outputTensor.Untyped())
	case HashKeccak256:
		err = cryptoOps.Keccak256(inputTensor.Untyped(), outputTensor.Untyped())
	case HashBlake3:
		// Blake3 not directly supported, use SHA256 as fallback via CPU
		return hashCPU(hashType, inputs)
	case HashPoseidon:
		err = cryptoOps.Poseidon(inputTensor.Untyped(), outputTensor.Untyped())
	default:
		return hashCPU(hashType, inputs)
	}

	if err != nil {
		return hashCPU(hashType, inputs)
	}

	// Sync GPU operations
	if err := sess.Sync(); err != nil {
		return hashCPU(hashType, inputs)
	}

	// Read results
	outputBytes, err := outputTensor.ToSlice()
	if err != nil {
		return hashCPU(hashType, inputs)
	}

	// Convert to [][32]byte
	results := make([][32]byte, n)
	for i := 0; i < n; i++ {
		copy(results[i][:], outputBytes[i*32:(i+1)*32])
	}

	return results, nil
}

func msmGPU(sess *accel.Session, curve Curve, scalars, points [][]byte) ([]byte, error) {
	n := len(scalars)
	if n == 0 {
		return nil, ErrInvalidInput
	}

	const scalarSize = 32
	pointSize := curve.pointSize()
	switch curve {
	case CurveSecp256k1, CurveBN254, CurveBLS12_381, CurveBanderwagon:
		// supported
	default:
		return msmCPU(curve, scalars, points)
	}

	// Pack scalars
	scalarBytes := make([]byte, n*scalarSize)
	for i, s := range scalars {
		if len(s) >= scalarSize {
			copy(scalarBytes[i*scalarSize:(i+1)*scalarSize], s[:scalarSize])
		} else {
			copy(scalarBytes[i*scalarSize:i*scalarSize+len(s)], s)
		}
	}

	// Pack points
	pointBytes := make([]byte, n*pointSize)
	for i, p := range points {
		if len(p) >= pointSize {
			copy(pointBytes[i*pointSize:(i+1)*pointSize], p[:pointSize])
		} else {
			copy(pointBytes[i*pointSize:i*pointSize+len(p)], p)
		}
	}

	// Create tensors
	scalarTensor, err := accel.NewTensorWithData[uint8](sess, []int{n, scalarSize}, scalarBytes)
	if err != nil {
		return msmCPU(curve, scalars, points)
	}
	defer scalarTensor.Close()

	pointTensor, err := accel.NewTensorWithData[uint8](sess, []int{n, pointSize}, pointBytes)
	if err != nil {
		return msmCPU(curve, scalars, points)
	}
	defer pointTensor.Close()

	// Result tensor (single point)
	resultTensor, err := accel.NewTensor[uint8](sess, []int{pointSize})
	if err != nil {
		return msmCPU(curve, scalars, points)
	}
	defer resultTensor.Close()

	// Dispatch MSM kernel via ZK ops
	err = sess.ZK().MSM(
		scalarTensor.Untyped(),
		pointTensor.Untyped(),
		resultTensor.Untyped(),
	)
	if err != nil {
		return msmCPU(curve, scalars, points)
	}

	// Sync
	if err := sess.Sync(); err != nil {
		return msmCPU(curve, scalars, points)
	}

	// Read result
	result, err := resultTensor.ToSlice()
	if err != nil {
		return msmCPU(curve, scalars, points)
	}

	return result, nil
}

func aggregateSigsGPU(sess *accel.Session, sigs [][]byte) ([]byte, error) {
	n := len(sigs)
	if n == 0 {
		return nil, ErrBatchEmpty
	}

	// BLS signature is 96 bytes (G2 point)
	const sigSize = 96

	// Pack signatures
	sigBytes := make([]byte, n*sigSize)
	for i, sig := range sigs {
		if len(sig) >= sigSize {
			copy(sigBytes[i*sigSize:(i+1)*sigSize], sig[:sigSize])
		} else {
			copy(sigBytes[i*sigSize:i*sigSize+len(sig)], sig)
		}
	}

	// Create tensors
	sigTensor, err := accel.NewTensorWithData[uint8](sess, []int{n, sigSize}, sigBytes)
	if err != nil {
		return aggregateSigsCPU(sigs)
	}
	defer sigTensor.Close()

	// Result tensor (single aggregated signature)
	resultTensor, err := accel.NewTensor[uint8](sess, []int{sigSize})
	if err != nil {
		return aggregateSigsCPU(sigs)
	}
	defer resultTensor.Close()

	// Dispatch BLS aggregation kernel
	err = sess.Crypto().BLSAggregate(sigTensor.Untyped(), resultTensor.Untyped())
	if err != nil {
		return aggregateSigsCPU(sigs)
	}

	// Sync
	if err := sess.Sync(); err != nil {
		return aggregateSigsCPU(sigs)
	}

	// Read result
	result, err := resultTensor.ToSlice()
	if err != nil {
		return aggregateSigsCPU(sigs)
	}

	return result, nil
}

func aggregatePksGPU(sess *accel.Session, pks [][]byte) ([]byte, error) {
	n := len(pks)
	if n == 0 {
		return nil, ErrBatchEmpty
	}

	// BLS public key is 48 bytes (G1 point)
	const pkSize = 48

	// Pack public keys
	pkBytes := make([]byte, n*pkSize)
	for i, pk := range pks {
		if len(pk) >= pkSize {
			copy(pkBytes[i*pkSize:(i+1)*pkSize], pk[:pkSize])
		} else {
			copy(pkBytes[i*pkSize:i*pkSize+len(pk)], pk)
		}
	}

	// Create tensors
	pkTensor, err := accel.NewTensorWithData[uint8](sess, []int{n, pkSize}, pkBytes)
	if err != nil {
		return aggregatePksCPU(pks)
	}
	defer pkTensor.Close()

	// Result tensor (single aggregated public key)
	resultTensor, err := accel.NewTensor[uint8](sess, []int{pkSize})
	if err != nil {
		return aggregatePksCPU(pks)
	}
	defer resultTensor.Close()

	// Use MSM for public key aggregation: sum(pk_i) where all scalars are 1
	// This is equivalent to adding all G1 points
	// Create all-ones scalars
	ones := make([]byte, n*32)
	for i := 0; i < n; i++ {
		ones[i*32] = 1 // scalar = 1 in little-endian
	}

	onesTensor, err := accel.NewTensorWithData[uint8](sess, []int{n, 32}, ones)
	if err != nil {
		return aggregatePksCPU(pks)
	}
	defer onesTensor.Close()

	// Use MSM: result = sum(1 * pk_i) = sum(pk_i)
	err = sess.ZK().MSM(onesTensor.Untyped(), pkTensor.Untyped(), resultTensor.Untyped())
	if err != nil {
		return aggregatePksCPU(pks)
	}

	// Sync
	if err := sess.Sync(); err != nil {
		return aggregatePksCPU(pks)
	}

	// Read result
	result, err := resultTensor.ToSlice()
	if err != nil {
		return aggregatePksCPU(pks)
	}

	return result, nil
}
