// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package code provides GPU-accelerated code-based cryptography
// operations. The flagship surface is HQC (Hamming Quasi-Cyclic), the
// NIST PQC Round-4 selected code-based KEM (NIST IR 8528, March 2025).
//
// HQC is family-disjoint from ML-KEM: a structural break against
// Module-LWE does not compromise the hardness assumption (Syndrome
// Decoding Problem) underpinning HQC. The crypto/hqc package wires
// this batch surface as its GPU dispatch path; CPU fallback always
// delegates to the PQClean reference under crypto/hqc/pqclean/.
//
// Three parameter sets:
//
//	HQC128 — NIST PQ Category 1 (~AES-128). pk=2249, sk=2305, ct=4433.
//	HQC192 — NIST PQ Category 3 (~AES-192). pk=4522, sk=4586, ct=8978.
//	HQC256 — NIST PQ Category 5 (~AES-256). pk=7245, sk=7317, ct=14421.
//
// All shared secrets are 64 bytes.
//
// The batch entry points consume a contiguous seed buffer where slot
// i's seed occupies seeds[i*SeedSize : (i+1)*SeedSize]. Replaying the
// same seed slice produces byte-identical output (FIPS-style
// determinism property — load-bearing for the on-chain HQC precompile).
package code

import (
	"errors"

	"github.com/luxfi/accel"
)

// Mode selects the HQC parameter set. Identical bit pattern to the
// C++ enum LuxHQCMode in luxfi/mlx/include/lux/gpu/hqc.h.
type Mode int

const (
	HQC128 Mode = 0
	HQC192 Mode = 1
	HQC256 Mode = 2
)

// Params returns the canonical byte sizes for each parameter set.
// Values are NIST-fixed; do not modify.
type Params struct {
	PublicKey    int
	SecretKey    int
	Ciphertext   int
	SharedSecret int
	// Seed sizes for the batch entry points.
	SeedKeypair int
	SeedEncaps  int
}

// ParamsFor returns the canonical Params for `mode`. Panics on
// unknown mode (callers should constrain Mode at the type boundary).
func ParamsFor(mode Mode) Params {
	switch mode {
	case HQC128:
		return Params{
			PublicKey: 2249, SecretKey: 2305,
			Ciphertext: 4433, SharedSecret: 64,
			SeedKeypair: 112, SeedEncaps: 48,
		}
	case HQC192:
		return Params{
			PublicKey: 4522, SecretKey: 4586,
			Ciphertext: 8978, SharedSecret: 64,
			SeedKeypair: 112, SeedEncaps: 48,
		}
	case HQC256:
		return Params{
			PublicKey: 7245, SecretKey: 7317,
			Ciphertext: 14421, SharedSecret: 64,
			SeedKeypair: 112, SeedEncaps: 48,
		}
	default:
		panic("code: unknown HQC mode")
	}
}

// Sentinel errors.
var (
	ErrInvalidMode       = errors.New("code: invalid HQC mode")
	ErrInvalidInput      = errors.New("code: invalid input")
	ErrSeedExhausted     = errors.New("code: seed buffer exhausted (RNG failed)")
	ErrCountZero         = errors.New("code: count must be > 0")
	ErrBufferSizeInvalid = errors.New("code: buffer size does not match count * per-slot size")
)

// HQCKeypairBatch generates `count` independent HQC keypairs of the
// given parameter set. The seeds buffer is consumed deterministically:
// slot i reads from seeds[i*p.SeedKeypair : (i+1)*p.SeedKeypair].
//
// pks and sks are output buffers, pre-allocated to:
//
//	len(pks) == count * p.PublicKey
//	len(sks) == count * p.SecretKey
//	len(seeds) == count * p.SeedKeypair
//
// Returns ErrSeedExhausted if any slot's PRNG ran out mid-op (which
// implies the caller under-provisioned the seed buffer).
func HQCKeypairBatch(mode Mode, pks, sks, seeds []byte, count int) error {
	if count <= 0 {
		return ErrCountZero
	}
	p := ParamsFor(mode)
	if len(pks) != count*p.PublicKey {
		return ErrBufferSizeInvalid
	}
	if len(sks) != count*p.SecretKey {
		return ErrBufferSizeInvalid
	}
	if len(seeds) != count*p.SeedKeypair {
		return ErrBufferSizeInvalid
	}
	sess, err := accel.NewSession()
	if err != nil {
		return hqcKeypairCPU(mode, pks, sks, seeds, count)
	}
	defer sess.Close()
	return hqcKeypairGPU(sess, mode, pks, sks, seeds, count)
}

// HQCEncapsBatch performs `count` independent encapsulations.
//
//	len(pks)  == count * p.PublicKey
//	len(cts)  == count * p.Ciphertext
//	len(sss)  == count * p.SharedSecret
//	len(seeds) == count * p.SeedEncaps
func HQCEncapsBatch(mode Mode, cts, sss, pks, seeds []byte, count int) error {
	if count <= 0 {
		return ErrCountZero
	}
	p := ParamsFor(mode)
	if len(pks) != count*p.PublicKey {
		return ErrBufferSizeInvalid
	}
	if len(cts) != count*p.Ciphertext {
		return ErrBufferSizeInvalid
	}
	if len(sss) != count*p.SharedSecret {
		return ErrBufferSizeInvalid
	}
	if len(seeds) != count*p.SeedEncaps {
		return ErrBufferSizeInvalid
	}
	sess, err := accel.NewSession()
	if err != nil {
		return hqcEncapsCPU(mode, cts, sss, pks, seeds, count)
	}
	defer sess.Close()
	return hqcEncapsGPU(sess, mode, cts, sss, pks, seeds, count)
}

// HQCDecapsBatch performs `count` independent decapsulations.
//
//	len(sks) == count * p.SecretKey
//	len(cts) == count * p.Ciphertext
//	len(sss) == count * p.SharedSecret
//
// Implicit rejection: a tampered ciphertext does NOT error — instead
// the corresponding sss slot receives a pseudorandom 64 bytes derived
// from (sk, ct). The caller compares against the counterparty's
// expected secret to detect rejection.
func HQCDecapsBatch(mode Mode, sss, cts, sks []byte, count int) error {
	if count <= 0 {
		return ErrCountZero
	}
	p := ParamsFor(mode)
	if len(sks) != count*p.SecretKey {
		return ErrBufferSizeInvalid
	}
	if len(cts) != count*p.Ciphertext {
		return ErrBufferSizeInvalid
	}
	if len(sss) != count*p.SharedSecret {
		return ErrBufferSizeInvalid
	}
	sess, err := accel.NewSession()
	if err != nil {
		return hqcDecapsCPU(mode, sss, cts, sks, count)
	}
	defer sess.Close()
	return hqcDecapsGPU(sess, mode, sss, cts, sks, count)
}

// GF2PolymulBatch multiplies pairs of GF(2)^N polynomials, one pair
// per slot, modulo X^N - 1. N is fixed by the parameter set:
//
//	HQC128: N = 17669 bits  (vec_n_size_64 = 277 uint64)
//	HQC192: N = 35851 bits  (vec_n_size_64 = 561 uint64)
//	HQC256: N = 57637 bits  (vec_n_size_64 = 901 uint64)
//
// Inputs and outputs are little-endian uint64 arrays. Slot i reads
// a[i*vecN:(i+1)*vecN] and b[i*vecN:(i+1)*vecN], writes
// c[i*vecN:(i+1)*vecN].
func GF2PolymulBatch(mode Mode, c, a, b []uint64, count int) error {
	if count <= 0 {
		return ErrCountZero
	}
	vecN, err := vecNSize64(mode)
	if err != nil {
		return err
	}
	if len(a) != count*vecN || len(b) != count*vecN || len(c) != count*vecN {
		return ErrBufferSizeInvalid
	}
	sess, err := accel.NewSession()
	if err != nil {
		return gf2PolymulCPU(mode, c, a, b, count)
	}
	defer sess.Close()
	return gf2PolymulGPU(sess, mode, c, a, b, count)
}

// ReedSolomonDecodeBatch decodes `count` independent Reed-Solomon
// codewords (PARAM_N1 bytes each) into PARAM_K-byte messages.
//
//	HQC128: PARAM_N1 = 46,  PARAM_K = 16
//	HQC192: PARAM_N1 = 56,  PARAM_K = 24
//	HQC256: PARAM_N1 = 90,  PARAM_K = 32
func ReedSolomonDecodeBatch(mode Mode, msgs, cdws []byte, count int) error {
	if count <= 0 {
		return ErrCountZero
	}
	n1, k, err := rsDims(mode)
	if err != nil {
		return err
	}
	if len(cdws) != count*n1 || len(msgs) != count*k {
		return ErrBufferSizeInvalid
	}
	sess, err := accel.NewSession()
	if err != nil {
		return rsDecodeCPU(mode, msgs, cdws, count)
	}
	defer sess.Close()
	return rsDecodeGPU(sess, mode, msgs, cdws, count)
}

// vecNSize64 returns the per-slot uint64 count for GF(2)^N polymul.
func vecNSize64(mode Mode) (int, error) {
	switch mode {
	case HQC128:
		return (17669 + 63) / 64, nil
	case HQC192:
		return (35851 + 63) / 64, nil
	case HQC256:
		return (57637 + 63) / 64, nil
	default:
		return 0, ErrInvalidMode
	}
}

// rsDims returns (PARAM_N1, PARAM_K) for the Reed-Solomon code.
func rsDims(mode Mode) (int, int, error) {
	switch mode {
	case HQC128:
		return 46, 16, nil
	case HQC192:
		return 56, 24, nil
	case HQC256:
		return 90, 32, nil
	default:
		return 0, 0, ErrInvalidMode
	}
}
