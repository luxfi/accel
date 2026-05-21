// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package hqc is the public accel surface for HQC (Hamming Quasi-Cyclic)
// post-quantum KEM operations.
//
// HQC is the NIST PQC Round-4 selected code-based KEM (NIST IR 8528,
// March 2025). Family-disjoint from ML-KEM: a structural break against
// MLWE does NOT compromise HQC, and vice-versa.
//
// This package is a thin re-export of accel/ops/code, named for the
// HQC primitive rather than the cryptographic family. The split exists
// so callers can write
//
//	import "github.com/luxfi/accel/hqc"
//
// and have hqc.Mode / hqc.KeypairBatch / hqc.EncapsBatch / etc. line up
// with crypto/hqc.Mode and crypto/hqc.* without a "code" layer of
// indirection in import paths.
//
// There is exactly one canonical home for each concern:
//
//	luxcpp/gpu/src/hqc_*.cpp      <-- C++ canonical kernels (Metal/CUDA/CPU)
//	luxcpp/gpu/include/lux/gpu/hqc.h  <-- C ABI
//	luxfi/accel/ops/code/         <-- Go binding to the C ABI
//	luxfi/accel/hqc/  (this pkg)  <-- HQC-named re-export
//	luxfi/gpu/hqc_*.go            <-- direct binding for low-level ops
//	luxfi/crypto/hqc/             <-- consumer Go package
//
// No primitive lives in more than one place. Re-exports are aliases,
// not new implementations.
package hqc

import (
	"github.com/luxfi/accel/ops/code"
)

// Mode aliases the underlying parameter-set enum.
type Mode = code.Mode

const (
	// HQC128 — NIST PQ Category 1 (~AES-128).
	HQC128 = code.HQC128
	// HQC192 — NIST PQ Category 3 (~AES-192).
	HQC192 = code.HQC192
	// HQC256 — NIST PQ Category 5 (~AES-256).
	HQC256 = code.HQC256
)

// Params aliases the byte-size descriptor.
type Params = code.Params

// ParamsFor returns the canonical sizes for mode.
func ParamsFor(mode Mode) Params { return code.ParamsFor(mode) }

// Error sentinels re-exported.
var (
	ErrInvalidMode       = code.ErrInvalidMode
	ErrInvalidInput      = code.ErrInvalidInput
	ErrSeedExhausted     = code.ErrSeedExhausted
	ErrCountZero         = code.ErrCountZero
	ErrBufferSizeInvalid = code.ErrBufferSizeInvalid
)

// KeypairBatch generates count independent HQC keypairs. See
// code.HQCKeypairBatch for layout requirements.
func KeypairBatch(mode Mode, pks, sks, seeds []byte, count int) error {
	return code.HQCKeypairBatch(mode, pks, sks, seeds, count)
}

// EncapsBatch performs count independent encapsulations.
func EncapsBatch(mode Mode, cts, sss, pks, seeds []byte, count int) error {
	return code.HQCEncapsBatch(mode, cts, sss, pks, seeds, count)
}

// DecapsBatch performs count independent decapsulations. HQC uses
// Hofheinz-Hövelmanns-Kiltz implicit rejection: a tampered ciphertext
// yields a pseudorandom shared secret rather than an error.
func DecapsBatch(mode Mode, sss, cts, sks []byte, count int) error {
	return code.HQCDecapsBatch(mode, sss, cts, sks, count)
}

// GF2PolymulBatch multiplies pairs of GF(2)^N polynomials mod (X^n - 1).
// Used inside HQC encapsulation (PKE) and decapsulation (FO-transform
// re-encryption). The lib is bit-sliced Karatsuba, identical to PQClean.
func GF2PolymulBatch(mode Mode, c, a, b []uint64, count int) error {
	return code.GF2PolymulBatch(mode, c, a, b, count)
}

// ReedSolomonDecodeBatch decodes count RS codewords (constant-time
// Berlekamp).
func ReedSolomonDecodeBatch(mode Mode, msgs, cdws []byte, count int) error {
	return code.ReedSolomonDecodeBatch(mode, msgs, cdws, count)
}
