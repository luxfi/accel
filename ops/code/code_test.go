// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package code

import (
	"bytes"
	"crypto/rand"
	"testing"
)

// helper — fill a byte slice with deterministic-ish bytes derived
// from `seed`. Used for property tests where we need reproducible
// inputs without spending entropy.
func detFill(buf []byte, seed byte) {
	for i := range buf {
		buf[i] = byte(int(seed) + (i * 31))
	}
}

// TestParams validates that the NIST-fixed sizes round-trip.
func TestParams(t *testing.T) {
	cases := []struct {
		mode     Mode
		name     string
		pk, sk   int
		ct, ss   int
		sk_seed  int
		enc_seed int
	}{
		{HQC128, "HQC-128", 2249, 2305, 4433, 64, 112, 48},
		{HQC192, "HQC-192", 4522, 4586, 8978, 64, 112, 48},
		{HQC256, "HQC-256", 7245, 7317, 14421, 64, 112, 48},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := ParamsFor(tc.mode)
			if p.PublicKey != tc.pk || p.SecretKey != tc.sk ||
				p.Ciphertext != tc.ct || p.SharedSecret != tc.ss ||
				p.SeedKeypair != tc.sk_seed || p.SeedEncaps != tc.enc_seed {
				t.Fatalf("%s params mismatch: got %+v", tc.name, p)
			}
		})
	}
}

// TestRoundTripSingle: keygen -> encap -> decap, verify the shared
// secret matches. Runs for all three parameter sets.
func TestRoundTripSingle(t *testing.T) {
	modes := []struct {
		mode Mode
		name string
	}{
		{HQC128, "HQC-128"},
		{HQC192, "HQC-192"},
		{HQC256, "HQC-256"},
	}
	for _, m := range modes {
		t.Run(m.name, func(t *testing.T) {
			p := ParamsFor(m.mode)

			seedKG := make([]byte, p.SeedKeypair)
			seedEnc := make([]byte, p.SeedEncaps)
			if _, err := rand.Read(seedKG); err != nil {
				t.Fatal(err)
			}
			if _, err := rand.Read(seedEnc); err != nil {
				t.Fatal(err)
			}

			pk := make([]byte, p.PublicKey)
			sk := make([]byte, p.SecretKey)
			if err := HQCKeypairBatch(m.mode, pk, sk, seedKG, 1); err != nil {
				t.Fatalf("keypair: %v", err)
			}

			ct := make([]byte, p.Ciphertext)
			ssEnc := make([]byte, p.SharedSecret)
			if err := HQCEncapsBatch(m.mode, ct, ssEnc, pk, seedEnc, 1); err != nil {
				t.Fatalf("encaps: %v", err)
			}

			ssDec := make([]byte, p.SharedSecret)
			if err := HQCDecapsBatch(m.mode, ssDec, ct, sk, 1); err != nil {
				t.Fatalf("decaps: %v", err)
			}

			if !bytes.Equal(ssEnc, ssDec) {
				t.Fatalf("shared secret mismatch: enc=%x dec=%x", ssEnc[:8], ssDec[:8])
			}
		})
	}
}

// TestDeterminism: same seed -> same key/ct/ss byte-for-byte. This
// is load-bearing for the on-chain HQC precompile (validators must
// reach consensus on encapsulation outputs).
func TestDeterminism(t *testing.T) {
	for _, mode := range []Mode{HQC128, HQC192, HQC256} {
		p := ParamsFor(mode)
		seedKG := make([]byte, p.SeedKeypair)
		seedEnc := make([]byte, p.SeedEncaps)
		detFill(seedKG, 0x42)
		detFill(seedEnc, 0x99)

		pk1 := make([]byte, p.PublicKey)
		pk2 := make([]byte, p.PublicKey)
		sk1 := make([]byte, p.SecretKey)
		sk2 := make([]byte, p.SecretKey)
		_ = HQCKeypairBatch(mode, pk1, sk1, seedKG, 1)
		_ = HQCKeypairBatch(mode, pk2, sk2, seedKG, 1)

		if !bytes.Equal(pk1, pk2) || !bytes.Equal(sk1, sk2) {
			t.Fatalf("mode=%v: same seed produced different keypairs", mode)
		}

		ct1 := make([]byte, p.Ciphertext)
		ct2 := make([]byte, p.Ciphertext)
		ss1 := make([]byte, 64)
		ss2 := make([]byte, 64)
		_ = HQCEncapsBatch(mode, ct1, ss1, pk1, seedEnc, 1)
		_ = HQCEncapsBatch(mode, ct2, ss2, pk1, seedEnc, 1)

		if !bytes.Equal(ct1, ct2) || !bytes.Equal(ss1, ss2) {
			t.Fatalf("mode=%v: same (pk, seed) produced different encaps", mode)
		}
	}
}

// TestBatchParity: a batch of N independent ops produces byte-equal
// output vs N sequential single-op calls. Verifies the TLS seed
// isolation inside the parallel C++ kernel.
func TestBatchParity(t *testing.T) {
	const N = 16
	for _, mode := range []Mode{HQC128, HQC192, HQC256} {
		p := ParamsFor(mode)

		seedsKG := make([]byte, N*p.SeedKeypair)
		seedsEnc := make([]byte, N*p.SeedEncaps)
		if _, err := rand.Read(seedsKG); err != nil {
			t.Fatal(err)
		}
		if _, err := rand.Read(seedsEnc); err != nil {
			t.Fatal(err)
		}

		// Batched.
		batchPks := make([]byte, N*p.PublicKey)
		batchSks := make([]byte, N*p.SecretKey)
		batchCts := make([]byte, N*p.Ciphertext)
		batchSS := make([]byte, N*p.SharedSecret)
		batchSSDec := make([]byte, N*p.SharedSecret)

		if err := HQCKeypairBatch(mode, batchPks, batchSks, seedsKG, N); err != nil {
			t.Fatalf("batch keypair: %v", err)
		}
		if err := HQCEncapsBatch(mode, batchCts, batchSS, batchPks, seedsEnc, N); err != nil {
			t.Fatalf("batch encaps: %v", err)
		}
		if err := HQCDecapsBatch(mode, batchSSDec, batchCts, batchSks, N); err != nil {
			t.Fatalf("batch decaps: %v", err)
		}

		// Sequential ground truth.
		seqPks := make([]byte, N*p.PublicKey)
		seqSks := make([]byte, N*p.SecretKey)
		seqCts := make([]byte, N*p.Ciphertext)
		seqSS := make([]byte, N*p.SharedSecret)
		for i := 0; i < N; i++ {
			pkI := seqPks[i*p.PublicKey : (i+1)*p.PublicKey]
			skI := seqSks[i*p.SecretKey : (i+1)*p.SecretKey]
			ctI := seqCts[i*p.Ciphertext : (i+1)*p.Ciphertext]
			ssI := seqSS[i*p.SharedSecret : (i+1)*p.SharedSecret]
			seedKgI := seedsKG[i*p.SeedKeypair : (i+1)*p.SeedKeypair]
			seedEncI := seedsEnc[i*p.SeedEncaps : (i+1)*p.SeedEncaps]
			if err := HQCKeypairBatch(mode, pkI, skI, seedKgI, 1); err != nil {
				t.Fatalf("seq keypair[%d]: %v", i, err)
			}
			if err := HQCEncapsBatch(mode, ctI, ssI, pkI, seedEncI, 1); err != nil {
				t.Fatalf("seq encaps[%d]: %v", i, err)
			}
		}

		if !bytes.Equal(batchPks, seqPks) {
			t.Fatalf("mode=%v: batch pks diverged from sequential", mode)
		}
		if !bytes.Equal(batchSks, seqSks) {
			t.Fatalf("mode=%v: batch sks diverged from sequential", mode)
		}
		if !bytes.Equal(batchCts, seqCts) {
			t.Fatalf("mode=%v: batch cts diverged from sequential", mode)
		}
		if !bytes.Equal(batchSS, seqSS) {
			t.Fatalf("mode=%v: batch shared secrets diverged from sequential", mode)
		}

		// Decap recovery — every slot's decap'd secret must equal the
		// encap'd secret.
		for i := 0; i < N; i++ {
			enc := batchSS[i*64 : (i+1)*64]
			dec := batchSSDec[i*64 : (i+1)*64]
			if !bytes.Equal(enc, dec) {
				t.Fatalf("mode=%v slot %d: decap mismatch", mode, i)
			}
		}
	}
}

// TestImplicitRejection: a flipped ciphertext byte yields a different
// (but valid) shared secret. The KEM does NOT error on tampering;
// the caller distinguishes by comparing secrets.
func TestImplicitRejection(t *testing.T) {
	mode := HQC128
	p := ParamsFor(mode)

	seedKG := make([]byte, p.SeedKeypair)
	seedEnc := make([]byte, p.SeedEncaps)
	_, _ = rand.Read(seedKG)
	_, _ = rand.Read(seedEnc)

	pk := make([]byte, p.PublicKey)
	sk := make([]byte, p.SecretKey)
	_ = HQCKeypairBatch(mode, pk, sk, seedKG, 1)

	ct := make([]byte, p.Ciphertext)
	ssEnc := make([]byte, 64)
	_ = HQCEncapsBatch(mode, ct, ssEnc, pk, seedEnc, 1)

	// Tamper.
	ctBad := make([]byte, p.Ciphertext)
	copy(ctBad, ct)
	ctBad[100] ^= 0xAA

	ssDecBad := make([]byte, 64)
	if err := HQCDecapsBatch(mode, ssDecBad, ctBad, sk, 1); err != nil {
		t.Fatalf("decap of tampered ct errored: %v (should be implicit reject)", err)
	}
	if bytes.Equal(ssEnc, ssDecBad) {
		t.Fatalf("tampered ct produced matching shared secret — implicit reject broken")
	}
}

// TestGF2PolymulSanity: zero polynomial multiplied with anything is
// zero; the kernel is commutative.
func TestGF2PolymulSanity(t *testing.T) {
	for _, mode := range []Mode{HQC128, HQC192, HQC256} {
		vecN, err := vecNSize64(mode)
		if err != nil {
			t.Fatal(err)
		}
		a := make([]uint64, vecN)
		b := make([]uint64, vecN)
		zero := make([]uint64, vecN)
		c1 := make([]uint64, vecN)
		c2 := make([]uint64, vecN)

		// Use deterministic values; mask off bits above PARAM_N so
		// the input vectors are in the valid representation domain.
		for i := 0; i < vecN; i++ {
			a[i] = uint64(0x1234567890ABCDEF) + uint64(i)*7
			b[i] = uint64(0xFEDCBA0987654321) + uint64(i)*13
		}
		// Mask high bits per mode (RED_MASK for HQC-128/192/256).
		switch mode {
		case HQC128:
			a[vecN-1] &= 0x1F
			b[vecN-1] &= 0x1F
		case HQC192:
			a[vecN-1] &= 0x7FF
			b[vecN-1] &= 0x7FF
		case HQC256:
			a[vecN-1] &= 0x1FFFFFFFFF
			b[vecN-1] &= 0x1FFFFFFFFF
		}

		// polymul(a, 0) == 0
		if err := GF2PolymulBatch(mode, c1, a, zero, 1); err != nil {
			t.Fatalf("mode=%v polymul a*0: %v", mode, err)
		}
		for _, x := range c1 {
			if x != 0 {
				t.Fatalf("mode=%v: polymul(a, 0) != 0", mode)
			}
		}

		// commutativity: a*b == b*a
		_ = GF2PolymulBatch(mode, c1, a, b, 1)
		_ = GF2PolymulBatch(mode, c2, b, a, 1)
		for i := range c1 {
			if c1[i] != c2[i] {
				t.Fatalf("mode=%v: polymul not commutative at limb %d", mode, i)
			}
		}
	}
}

// TestArgumentValidation: catches buffer size mismatches.
func TestArgumentValidation(t *testing.T) {
	p := ParamsFor(HQC128)

	// count = 0
	if err := HQCKeypairBatch(HQC128, nil, nil, nil, 0); err != ErrCountZero {
		t.Fatalf("count=0 should be ErrCountZero, got %v", err)
	}

	// undersized buffers.
	pks := make([]byte, p.PublicKey-1)
	sks := make([]byte, p.SecretKey)
	seeds := make([]byte, p.SeedKeypair)
	if err := HQCKeypairBatch(HQC128, pks, sks, seeds, 1); err != ErrBufferSizeInvalid {
		t.Fatalf("undersized pks should be ErrBufferSizeInvalid, got %v", err)
	}
}

// BenchmarkBatch measures throughput at various batch sizes. Useful
// for verifying that the OpenMP parallel loop in lux_hqc_*_batch
// scales as expected on the host's core count.
func BenchmarkEncapsHQC128(b *testing.B) {
	benchmarkEncaps(b, HQC128, 32)
}

func BenchmarkEncapsHQC192(b *testing.B) {
	benchmarkEncaps(b, HQC192, 32)
}

func BenchmarkEncapsHQC256(b *testing.B) {
	benchmarkEncaps(b, HQC256, 32)
}

func benchmarkEncaps(b *testing.B, mode Mode, batchN int) {
	p := ParamsFor(mode)
	pks := make([]byte, batchN*p.PublicKey)
	sks := make([]byte, batchN*p.SecretKey)
	seedsKG := make([]byte, batchN*p.SeedKeypair)
	seedsEnc := make([]byte, batchN*p.SeedEncaps)
	_, _ = rand.Read(seedsKG)
	_, _ = rand.Read(seedsEnc)
	if err := HQCKeypairBatch(mode, pks, sks, seedsKG, batchN); err != nil {
		b.Fatal(err)
	}

	cts := make([]byte, batchN*p.Ciphertext)
	sss := make([]byte, batchN*p.SharedSecret)

	b.ResetTimer()
	b.SetBytes(int64(batchN * p.Ciphertext))
	for i := 0; i < b.N; i++ {
		if err := HQCEncapsBatch(mode, cts, sss, pks, seedsEnc, batchN); err != nil {
			b.Fatal(err)
		}
	}
}
