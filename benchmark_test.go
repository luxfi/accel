// Copyright (C) 2025, Lux Industries, Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package accel

import (
	"crypto/rand"
	"errors"
	"testing"
)

// =============================================================================
// NTT Benchmarks
// =============================================================================

// generateNTTRoots generates primitive roots of unity for NTT.
// Uses a known prime modulus suitable for NTT operations.
func generateNTTRoots(n int) ([]uint64, []uint64, uint64) {
	// BN254 scalar field modulus (commonly used in ZK proofs)
	modulus := uint64(0xFFFFFFFF00000001) // 2^64 - 2^32 + 1 (Goldilocks field)

	// Generate test roots (in production these would be properly computed)
	roots := make([]uint64, n)
	invRoots := make([]uint64, n)
	for i := 0; i < n; i++ {
		roots[i] = uint64(i+1) % modulus
		invRoots[i] = uint64(n-i) % modulus
	}
	return roots, invRoots, modulus
}

// generateNTTCoeffs generates random polynomial coefficients.
func generateNTTCoeffs(n int, modulus uint64) []uint64 {
	coeffs := make([]uint64, n)
	for i := 0; i < n; i++ {
		var buf [8]byte
		rand.Read(buf[:])
		coeffs[i] = (uint64(buf[0]) | uint64(buf[1])<<8 | uint64(buf[2])<<16 | uint64(buf[3])<<24 |
			uint64(buf[4])<<32 | uint64(buf[5])<<40 | uint64(buf[6])<<48 | uint64(buf[7])<<56) % modulus
	}
	return coeffs
}

func BenchmarkNTTForward(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	sizes := []int{256, 1024, 4096, 8192}

	for _, size := range sizes {
		roots, _, modulus := generateNTTRoots(size)
		coeffs := generateNTTCoeffs(size, modulus)

		b.Run(formatSize("n=", size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * 8)) // 8 bytes per uint64

			for i := 0; i < b.N; i++ {
				// Make a copy to avoid modifying original
				input := make([]uint64, size)
				copy(input, coeffs)

				err := NTTForward(input, roots, modulus)
				if err != nil && !errors.Is(err, ErrNotSupported) {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkNTTInverse(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	sizes := []int{256, 1024, 4096, 8192}

	for _, size := range sizes {
		_, invRoots, modulus := generateNTTRoots(size)
		coeffs := generateNTTCoeffs(size, modulus)

		b.Run(formatSize("n=", size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * 8))

			for i := 0; i < b.N; i++ {
				input := make([]uint64, size)
				copy(input, coeffs)

				err := NTTInverse(input, invRoots, modulus)
				if err != nil && !errors.Is(err, ErrNotSupported) {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkNTTRoundtrip benchmarks forward followed by inverse NTT.
func BenchmarkNTTRoundtrip(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	sizes := []int{256, 1024, 4096, 8192}

	for _, size := range sizes {
		roots, invRoots, modulus := generateNTTRoots(size)
		coeffs := generateNTTCoeffs(size, modulus)

		b.Run(formatSize("n=", size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * 8 * 2)) // Two transforms

			for i := 0; i < b.N; i++ {
				input := make([]uint64, size)
				copy(input, coeffs)

				if err := NTTForward(input, roots, modulus); err != nil && !errors.Is(err, ErrNotSupported) {
					b.Fatal(err)
				}
				if err := NTTInverse(input, invRoots, modulus); err != nil && !errors.Is(err, ErrNotSupported) {
					b.Fatal(err)
				}
			}
		})
	}
}

// =============================================================================
// BLS Verification Benchmarks
// =============================================================================

// generateBLSTestData generates fake BLS signature data for benchmarking.
// In production, real cryptographic operations would be used.
func generateBLSTestData(n int) (pks, sigs, msgs [][]byte) {
	pks = make([][]byte, n)
	sigs = make([][]byte, n)
	msgs = make([][]byte, n)

	for i := 0; i < n; i++ {
		pks[i] = make([]byte, 48)  // G1 point (compressed)
		sigs[i] = make([]byte, 96) // G2 point (compressed)
		msgs[i] = make([]byte, 32) // Message hash

		rand.Read(pks[i])
		rand.Read(sigs[i])
		rand.Read(msgs[i])
	}
	return
}

func BenchmarkBLSVerifyBatch(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	batchSizes := []int{10, 100, 1000}

	for _, size := range batchSizes {
		// Skip if below threshold
		if size < BLSBatchVerifyThreshold {
			b.Run(formatSize("n=", size), func(b *testing.B) {
				b.Skip("batch size below GPU threshold")
			})
			continue
		}

		pks, sigs, msgs := generateBLSTestData(size)

		b.Run(formatSize("n=", size), func(b *testing.B) {
			b.ReportAllocs()
			// Data size: 48 (pk) + 96 (sig) + 32 (msg) = 176 bytes per signature
			b.SetBytes(int64(size * 176))

			for i := 0; i < b.N; i++ {
				_, err := BLSBatchVerify(pks, sigs, msgs)
				if err != nil && !errors.Is(err, ErrNotSupported) {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkBLSVerifyBatchScaling tests how verification scales with batch size.
func BenchmarkBLSVerifyBatchScaling(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	// Test scaling from threshold to large batches
	sizes := []int{64, 128, 256, 512, 1024, 2048, 4096}

	for _, size := range sizes {
		if size < BLSBatchVerifyThreshold {
			continue
		}

		pks, sigs, msgs := generateBLSTestData(size)

		b.Run(formatSize("n=", size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * 176))

			for i := 0; i < b.N; i++ {
				_, err := BLSBatchVerify(pks, sigs, msgs)
				if err != nil && !errors.Is(err, ErrNotSupported) {
					b.Fatal(err)
				}
			}
		})
	}
}

// =============================================================================
// SHA256 Batch Benchmarks
// =============================================================================

// generateHashInputs generates random inputs for hashing.
func generateHashInputs(n, inputSize int) [][]byte {
	inputs := make([][]byte, n)
	for i := 0; i < n; i++ {
		inputs[i] = make([]byte, inputSize)
		rand.Read(inputs[i])
	}
	return inputs
}

func BenchmarkSHA256Batch(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	batchSizes := []int{32, 100, 1000, 10000}
	inputSize := 64 // 64 bytes per input (typical message size)

	for _, size := range batchSizes {
		if size < HashBatchThreshold {
			b.Run(formatSize("n=", size), func(b *testing.B) {
				b.Skip("batch size below GPU threshold")
			})
			continue
		}

		inputs := generateHashInputs(size, inputSize)

		b.Run(formatSize("n=", size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * inputSize))

			for i := 0; i < b.N; i++ {
				_, err := SHA256Batch(inputs)
				if err != nil && !errors.Is(err, ErrNotSupported) {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkSHA256BatchVaryingInputSize tests SHA256 with different input sizes.
func BenchmarkSHA256BatchVaryingInputSize(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	batchSize := 1000
	inputSizes := []int{32, 64, 128, 256, 512, 1024}

	if batchSize < HashBatchThreshold {
		b.Skip("batch size below GPU threshold")
	}

	for _, inputSize := range inputSizes {
		inputs := generateHashInputs(batchSize, inputSize)

		b.Run(formatSize("input=", inputSize)+"bytes", func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(batchSize * inputSize))

			for i := 0; i < b.N; i++ {
				_, err := SHA256Batch(inputs)
				if err != nil && !errors.Is(err, ErrNotSupported) {
					b.Fatal(err)
				}
			}
		})
	}
}

// =============================================================================
// Keccak256 Batch Benchmarks
// =============================================================================

func BenchmarkKeccak256Batch(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	batchSizes := []int{32, 100, 1000, 10000}
	inputSize := 64

	for _, size := range batchSizes {
		if size < HashBatchThreshold {
			b.Run(formatSize("n=", size), func(b *testing.B) {
				b.Skip("batch size below GPU threshold")
			})
			continue
		}

		inputs := generateHashInputs(size, inputSize)

		b.Run(formatSize("n=", size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * inputSize))

			for i := 0; i < b.N; i++ {
				_, err := Keccak256Batch(inputs)
				if err != nil && !errors.Is(err, ErrNotSupported) {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkKeccak256VsSHA256 compares Keccak256 and SHA256 performance.
func BenchmarkKeccak256VsSHA256(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	batchSize := 1000
	inputSize := 64

	if batchSize < HashBatchThreshold {
		b.Skip("batch size below GPU threshold")
	}

	inputs := generateHashInputs(batchSize, inputSize)

	b.Run("SHA256", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(batchSize * inputSize))

		for i := 0; i < b.N; i++ {
			_, err := SHA256Batch(inputs)
			if err != nil && !errors.Is(err, ErrNotSupported) {
				b.Fatal(err)
			}
		}
	})

	b.Run("Keccak256", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(batchSize * inputSize))

		for i := 0; i < b.N; i++ {
			_, err := Keccak256Batch(inputs)
			if err != nil && !errors.Is(err, ErrNotSupported) {
				b.Fatal(err)
			}
		}
	})
}

// =============================================================================
// Tensor Benchmarks
// =============================================================================

func BenchmarkTensorCreate(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	session, err := DefaultSession()
	if err != nil {
		b.Fatal(err)
	}

	sizes := []struct {
		name  string
		shape []int
	}{
		{"small_1D", []int{256}},
		{"medium_1D", []int{4096}},
		{"large_1D", []int{65536}},
		{"small_2D", []int{64, 64}},
		{"medium_2D", []int{256, 256}},
		{"large_2D", []int{1024, 1024}},
		{"small_3D", []int{32, 32, 32}},
		{"medium_3D", []int{64, 64, 64}},
	}

	for _, tc := range sizes {
		totalBytes := 4 // float32
		for _, dim := range tc.shape {
			totalBytes *= dim
		}

		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(totalBytes))

			for i := 0; i < b.N; i++ {
				tensor, err := NewTensor[float32](session, tc.shape)
				if err != nil {
					b.Fatal(err)
				}
				tensor.Close()
			}
		})
	}
}

func BenchmarkTensorCreateWithData(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	session, err := DefaultSession()
	if err != nil {
		b.Fatal(err)
	}

	sizes := []int{256, 1024, 4096, 16384, 65536}

	for _, size := range sizes {
		data := make([]float32, size)
		for i := range data {
			data[i] = float32(i)
		}

		b.Run(formatSize("n=", size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * 4))

			for i := 0; i < b.N; i++ {
				tensor, err := NewTensorWithData[float32](session, []int{size}, data)
				if err != nil {
					b.Fatal(err)
				}
				tensor.Close()
			}
		})
	}
}

func BenchmarkTensorToSlice(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	session, err := DefaultSession()
	if err != nil {
		b.Fatal(err)
	}

	sizes := []int{256, 1024, 4096, 16384, 65536}

	for _, size := range sizes {
		tensor, err := NewTensor[float32](session, []int{size})
		if err != nil {
			b.Fatal(err)
		}

		b.Run(formatSize("n=", size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * 4))

			for i := 0; i < b.N; i++ {
				_, err := tensor.ToSlice()
				if err != nil {
					b.Fatal(err)
				}
			}
		})

		tensor.Close()
	}
}

func BenchmarkTensorFromSlice(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	session, err := DefaultSession()
	if err != nil {
		b.Fatal(err)
	}

	sizes := []int{256, 1024, 4096, 16384, 65536}

	for _, size := range sizes {
		tensor, err := NewTensor[float32](session, []int{size})
		if err != nil {
			b.Fatal(err)
		}

		data := make([]float32, size)
		for i := range data {
			data[i] = float32(i)
		}

		b.Run(formatSize("n=", size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * 4))

			for i := 0; i < b.N; i++ {
				err := tensor.FromSlice(data)
				if err != nil {
					b.Fatal(err)
				}
			}
		})

		tensor.Close()
	}
}

// =============================================================================
// Session Benchmarks
// =============================================================================

func BenchmarkSessionCreate(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	// First ensure library is initialized
	if err := Init(); err != nil {
		b.Fatal(err)
	}

	b.Run("NewSession", func(b *testing.B) {
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			session, err := NewSession()
			if err != nil {
				b.Fatal(err)
			}
			session.Close()
		}
	})

	// Test session creation with specific options
	b.Run("NewSession_WithAsync", func(b *testing.B) {
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			session, err := NewSession(WithAsync(true))
			if err != nil {
				b.Fatal(err)
			}
			session.Close()
		}
	})

	// Benchmark getting subsystem interfaces
	session, err := NewSession()
	if err != nil {
		b.Fatal(err)
	}
	defer session.Close()

	b.Run("Session_Crypto", func(b *testing.B) {
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_ = session.Crypto()
		}
	})

	b.Run("Session_ZK", func(b *testing.B) {
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_ = session.ZK()
		}
	})

	b.Run("Session_Lattice", func(b *testing.B) {
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_ = session.Lattice()
		}
	})
}

func BenchmarkSessionSync(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	session, err := NewSession()
	if err != nil {
		b.Fatal(err)
	}
	defer session.Close()

	b.Run("Sync_NoOps", func(b *testing.B) {
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			if err := session.Sync(); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// =============================================================================
// MSM Benchmarks
// =============================================================================

func generateMSMTestData(n int) (scalars, bases [][]byte) {
	scalars = make([][]byte, n)
	bases = make([][]byte, n)

	for i := 0; i < n; i++ {
		scalars[i] = make([]byte, 32) // 256-bit scalar
		bases[i] = make([]byte, 64)   // Affine point (x, y) on curve

		rand.Read(scalars[i])
		rand.Read(bases[i])
	}
	return
}

func BenchmarkMSM(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	batchSizes := []int{64, 256, 1024, 4096, 16384}

	for _, size := range batchSizes {
		if size < MSMBatchThreshold {
			b.Run(formatSize("n=", size), func(b *testing.B) {
				b.Skip("batch size below GPU threshold")
			})
			continue
		}

		scalars, bases := generateMSMTestData(size)

		b.Run(formatSize("n=", size), func(b *testing.B) {
			b.ReportAllocs()
			// Data size: 32 (scalar) + 64 (base) = 96 bytes per element
			b.SetBytes(int64(size * 96))

			for i := 0; i < b.N; i++ {
				_, err := MSM(scalars, bases)
				if err != nil && !errors.Is(err, ErrNotSupported) {
					b.Fatal(err)
				}
			}
		})
	}
}

// =============================================================================
// Merkle Tree Benchmarks
// =============================================================================

func generateMerkleLeaves(n int) [][]byte {
	leaves := make([][]byte, n)
	for i := 0; i < n; i++ {
		leaves[i] = make([]byte, 32)
		rand.Read(leaves[i])
	}
	return leaves
}

func BenchmarkMerkleRoot(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	// Merkle tree sizes (must be powers of 2 for typical implementations)
	sizes := []int{32, 64, 128, 256, 512, 1024, 4096}

	for _, size := range sizes {
		if size < HashBatchThreshold {
			b.Run(formatSize("leaves=", size), func(b *testing.B) {
				b.Skip("batch size below GPU threshold")
			})
			continue
		}

		leaves := generateMerkleLeaves(size)

		b.Run(formatSize("leaves=", size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * 32))

			for i := 0; i < b.N; i++ {
				_, err := MerkleRoot(leaves)
				if err != nil && !errors.Is(err, ErrNotSupported) {
					b.Fatal(err)
				}
			}
		})
	}
}

// =============================================================================
// Lattice Crypto Benchmarks
// =============================================================================

func BenchmarkKyberKeyGen(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	b.ReportAllocs()
	b.SetBytes(int64(KyberPublicKeySize + KyberSecretKeySize))

	for i := 0; i < b.N; i++ {
		_, _, err := KyberKeyGen()
		if err != nil && !errors.Is(err, ErrNotSupported) {
			b.Fatal(err)
		}
	}
}

func BenchmarkKyberEncapsDecaps(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	// Generate a key pair first
	pk, sk, err := KyberKeyGen()
	if errors.Is(err, ErrNotSupported) {
		b.Skip("Kyber not supported")
	}
	if err != nil {
		b.Fatal(err)
	}

	b.Run("Encaps", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(KyberCiphertextSize + KyberSharedKeySize))

		for i := 0; i < b.N; i++ {
			_, _, err := KyberEncaps(pk)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	// Generate a ciphertext for decapsulation benchmark
	ct, _, err := KyberEncaps(pk)
	if err != nil {
		b.Fatal(err)
	}

	b.Run("Decaps", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(KyberSharedKeySize))

		for i := 0; i < b.N; i++ {
			_, err := KyberDecaps(ct, sk)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkDilithiumSignVerify(b *testing.B) {
	if !Available() {
		b.Skip("GPU acceleration not available")
	}

	session, err := DefaultSession()
	if err != nil {
		b.Fatal(err)
	}

	// Generate key pair
	pkTensor, err := NewTensor[uint8](session, []int{DilithiumPublicKeySize})
	if err != nil {
		b.Fatal(err)
	}
	defer pkTensor.Close()

	skTensor, err := NewTensor[uint8](session, []int{DilithiumSecretKeySize})
	if err != nil {
		b.Fatal(err)
	}
	defer skTensor.Close()

	if err := session.Lattice().DilithiumKeyGen(pkTensor.Untyped(), skTensor.Untyped()); err != nil {
		if errors.Is(err, ErrNotSupported) {
			b.Skip("Dilithium not supported")
		}
		b.Fatal(err)
	}

	pk, err := pkTensor.ToSlice()
	if err != nil {
		b.Fatal(err)
	}
	sk, err := skTensor.ToSlice()
	if err != nil {
		b.Fatal(err)
	}

	msg := make([]byte, 64)
	rand.Read(msg)

	b.Run("Sign", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(DilithiumSignatureSize))

		for i := 0; i < b.N; i++ {
			_, err := DilithiumSign(msg, sk)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	// Generate signature for verification benchmark
	sig, err := DilithiumSign(msg, sk)
	if err != nil {
		b.Fatal(err)
	}

	b.Run("Verify", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(msg) + DilithiumSignatureSize + DilithiumPublicKeySize))

		for i := 0; i < b.N; i++ {
			_, err := DilithiumVerify(msg, sig, pk)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// =============================================================================
// Non-CGO Path Benchmarks
// =============================================================================

// BenchmarkNoCGOPath tests the overhead of the non-CGO stub implementations.
// These should be nearly zero-cost when GPU is not available.
func BenchmarkNoCGOPath(b *testing.B) {
	// These benchmarks intentionally run even without GPU to measure stub overhead

	b.Run("DefaultSession_Stub", func(b *testing.B) {
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_, _ = DefaultSession()
		}
	})

	b.Run("Available_Check", func(b *testing.B) {
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_ = Available()
		}
	})

	b.Run("Backends_List", func(b *testing.B) {
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_ = Backends()
		}
	})
}

// =============================================================================
// Helpers
// =============================================================================

func formatSize(prefix string, n int) string {
	if n >= 1000000 {
		return prefix + itoa(n/1000000) + "M"
	}
	if n >= 1000 {
		return prefix + itoa(n/1000) + "K"
	}
	return prefix + itoa(n)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var digits []byte
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}
