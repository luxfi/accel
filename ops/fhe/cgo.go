//go:build cgo && luxfhe

// Package fhe provides GPU-accelerated FHE operations via luxfhe C++ library.
//
// When built with `go build -tags luxfhe`, this uses the high-performance
// C++ implementation. Otherwise, falls back to pure Go.
//
// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
package fhe

/*
#cgo pkg-config: lux-fhe
#include <lux/fhe/c_api.h>
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

// Errors from C library
var (
	ErrNullPtr     = errors.New("fhe: null pointer")
	ErrInvalidArg  = errors.New("fhe: invalid argument")
	ErrAlloc       = errors.New("fhe: allocation failed")
	ErrKeyGen      = errors.New("fhe: key generation failed")
	ErrEncrypt     = errors.New("fhe: encryption failed")
	ErrDecrypt     = errors.New("fhe: decryption failed")
	ErrBootstrap   = errors.New("fhe: bootstrap failed")
	ErrGate        = errors.New("fhe: gate evaluation failed")
	ErrSerialize   = errors.New("fhe: serialization failed")
	ErrDeserialize = errors.New("fhe: deserialization failed")
	ErrNotInit     = errors.New("fhe: not initialized")
)

func cgoErr(e C.LuxFheError) error {
	switch e {
	case C.LUX_FHE_OK:
		return nil
	case C.LUX_FHE_ERR_NULL_PTR:
		return ErrNullPtr
	case C.LUX_FHE_ERR_INVALID_PARAM:
		return ErrInvalidArg
	case C.LUX_FHE_ERR_ALLOC:
		return ErrAlloc
	case C.LUX_FHE_ERR_KEYGEN:
		return ErrKeyGen
	case C.LUX_FHE_ERR_ENCRYPT:
		return ErrEncrypt
	case C.LUX_FHE_ERR_DECRYPT:
		return ErrDecrypt
	case C.LUX_FHE_ERR_BOOTSTRAP:
		return ErrBootstrap
	case C.LUX_FHE_ERR_GATE:
		return ErrGate
	case C.LUX_FHE_ERR_SERIALIZE:
		return ErrSerialize
	case C.LUX_FHE_ERR_DESERIALIZE:
		return ErrDeserialize
	case C.LUX_FHE_ERR_NOT_INIT:
		return ErrNotInit
	default:
		return errors.New("fhe: unknown error")
	}
}

// =============================================================================
// Parameter Sets
// =============================================================================

// CgoParams represents FHE parameter sets (C++ backend)
type CgoParams int

const (
	CgoParamsToy     CgoParams = C.LUX_FHE_PARAMS_TOY
	CgoParamsMedium  CgoParams = C.LUX_FHE_PARAMS_MEDIUM
	CgoParamsSTD128  CgoParams = C.LUX_FHE_PARAMS_STD128
	CgoParamsSTD128Q CgoParams = C.LUX_FHE_PARAMS_STD128Q
	CgoParamsSTD192  CgoParams = C.LUX_FHE_PARAMS_STD192
	CgoParamsSTD192Q CgoParams = C.LUX_FHE_PARAMS_STD192Q
	CgoParamsSTD256  CgoParams = C.LUX_FHE_PARAMS_STD256
	CgoParamsSTD256Q CgoParams = C.LUX_FHE_PARAMS_STD256Q
)

// CgoMethod represents bootstrapping methods
type CgoMethod int

const (
	CgoMethodAP      CgoMethod = C.LUX_FHE_METHOD_AP
	CgoMethodGINX    CgoMethod = C.LUX_FHE_METHOD_GINX
	CgoMethodLMKCDEY CgoMethod = C.LUX_FHE_METHOD_LMKCDEY
)

// =============================================================================
// Context
// =============================================================================

// CgoContext wraps the C++ FHE context
type CgoContext struct {
	ptr *C.LuxFheContext
}

// NewCgoContext creates a new FHE context with C++ backend
func NewCgoContext(params CgoParams, method CgoMethod) (*CgoContext, error) {
	var ptr *C.LuxFheContext
	err := C.lux_fhe_context_new(&ptr, C.LuxFheParams(params), C.LuxFheMethod(method))
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	ctx := &CgoContext{ptr: ptr}
	runtime.SetFinalizer(ctx, (*CgoContext).Free)
	return ctx, nil
}

// Free releases the context resources
func (c *CgoContext) Free() {
	if c.ptr != nil {
		C.lux_fhe_context_free(c.ptr)
		c.ptr = nil
	}
}

// N returns the LWE dimension
func (c *CgoContext) N() uint32 {
	return uint32(C.lux_fhe_context_n(c.ptr))
}

// RingDim returns the ring dimension
func (c *CgoContext) RingDim() uint32 {
	return uint32(C.lux_fhe_context_ring_dim(c.ptr))
}

// =============================================================================
// Keys
// =============================================================================

// CgoSecretKey wraps C++ secret key
type CgoSecretKey struct {
	ptr *C.LuxFheSecretKey
}

// CgoPublicKey wraps C++ public key
type CgoPublicKey struct {
	ptr *C.LuxFhePublicKey
}

// CgoBootstrapKey wraps C++ bootstrap key
type CgoBootstrapKey struct {
	ptr *C.LuxFheBootstrapKey
}

// GenSecretKey generates a secret key
func (c *CgoContext) GenSecretKey() (*CgoSecretKey, error) {
	var ptr *C.LuxFheSecretKey
	err := C.lux_fhe_keygen_secret(c.ptr, &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	sk := &CgoSecretKey{ptr: ptr}
	runtime.SetFinalizer(sk, (*CgoSecretKey).Free)
	return sk, nil
}

// GenPublicKey generates a public key from secret key
func (c *CgoContext) GenPublicKey(sk *CgoSecretKey) (*CgoPublicKey, error) {
	var ptr *C.LuxFhePublicKey
	err := C.lux_fhe_keygen_public(c.ptr, sk.ptr, &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	pk := &CgoPublicKey{ptr: ptr}
	runtime.SetFinalizer(pk, (*CgoPublicKey).Free)
	return pk, nil
}

// GenBootstrapKey generates bootstrap key for homomorphic gates
func (c *CgoContext) GenBootstrapKey(sk *CgoSecretKey) (*CgoBootstrapKey, error) {
	var ptr *C.LuxFheBootstrapKey
	err := C.lux_fhe_keygen_bootstrap(c.ptr, sk.ptr, &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	bsk := &CgoBootstrapKey{ptr: ptr}
	runtime.SetFinalizer(bsk, (*CgoBootstrapKey).Free)
	return bsk, nil
}

func (sk *CgoSecretKey) Free() {
	if sk.ptr != nil {
		C.lux_fhe_secretkey_free(sk.ptr)
		sk.ptr = nil
	}
}

func (pk *CgoPublicKey) Free() {
	if pk.ptr != nil {
		C.lux_fhe_publickey_free(pk.ptr)
		pk.ptr = nil
	}
}

func (bsk *CgoBootstrapKey) Free() {
	if bsk.ptr != nil {
		C.lux_fhe_bootstrapkey_free(bsk.ptr)
		bsk.ptr = nil
	}
}

// =============================================================================
// Ciphertext
// =============================================================================

// CgoCiphertext wraps C++ ciphertext
type CgoCiphertext struct {
	ptr *C.LuxFheCiphertext
}

// Encrypt encrypts a bit using secret key
func (c *CgoContext) Encrypt(sk *CgoSecretKey, plaintext bool) (*CgoCiphertext, error) {
	var ptr *C.LuxFheCiphertext
	err := C.lux_fhe_encrypt(c.ptr, sk.ptr, C.bool(plaintext), &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	ct := &CgoCiphertext{ptr: ptr}
	runtime.SetFinalizer(ct, (*CgoCiphertext).Free)
	return ct, nil
}

// EncryptPK encrypts a bit using public key
func (c *CgoContext) EncryptPK(pk *CgoPublicKey, plaintext bool) (*CgoCiphertext, error) {
	var ptr *C.LuxFheCiphertext
	err := C.lux_fhe_encrypt_pk(c.ptr, pk.ptr, C.bool(plaintext), &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	ct := &CgoCiphertext{ptr: ptr}
	runtime.SetFinalizer(ct, (*CgoCiphertext).Free)
	return ct, nil
}

// Decrypt decrypts a ciphertext
func (c *CgoContext) Decrypt(sk *CgoSecretKey, ct *CgoCiphertext) (bool, error) {
	var result C.bool
	err := C.lux_fhe_decrypt(c.ptr, sk.ptr, ct.ptr, &result)
	if err != C.LUX_FHE_OK {
		return false, cgoErr(err)
	}
	return bool(result), nil
}

func (ct *CgoCiphertext) Free() {
	if ct.ptr != nil {
		C.lux_fhe_ciphertext_free(ct.ptr)
		ct.ptr = nil
	}
}

// Clone creates a copy of the ciphertext
func (ct *CgoCiphertext) Clone() (*CgoCiphertext, error) {
	var ptr *C.LuxFheCiphertext
	err := C.lux_fhe_ciphertext_clone(ct.ptr, &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	clone := &CgoCiphertext{ptr: ptr}
	runtime.SetFinalizer(clone, (*CgoCiphertext).Free)
	return clone, nil
}

// =============================================================================
// Gates
// =============================================================================

// Not evaluates NOT gate (no bootstrapping needed)
func (c *CgoContext) Not(ct *CgoCiphertext) (*CgoCiphertext, error) {
	var ptr *C.LuxFheCiphertext
	err := C.lux_fhe_not(c.ptr, ct.ptr, &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	result := &CgoCiphertext{ptr: ptr}
	runtime.SetFinalizer(result, (*CgoCiphertext).Free)
	return result, nil
}

// And evaluates AND gate with bootstrapping
func (c *CgoContext) And(bsk *CgoBootstrapKey, a, b *CgoCiphertext) (*CgoCiphertext, error) {
	var ptr *C.LuxFheCiphertext
	err := C.lux_fhe_and(c.ptr, bsk.ptr, a.ptr, b.ptr, &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	result := &CgoCiphertext{ptr: ptr}
	runtime.SetFinalizer(result, (*CgoCiphertext).Free)
	return result, nil
}

// Or evaluates OR gate with bootstrapping
func (c *CgoContext) Or(bsk *CgoBootstrapKey, a, b *CgoCiphertext) (*CgoCiphertext, error) {
	var ptr *C.LuxFheCiphertext
	err := C.lux_fhe_or(c.ptr, bsk.ptr, a.ptr, b.ptr, &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	result := &CgoCiphertext{ptr: ptr}
	runtime.SetFinalizer(result, (*CgoCiphertext).Free)
	return result, nil
}

// Xor evaluates XOR gate with bootstrapping
func (c *CgoContext) Xor(bsk *CgoBootstrapKey, a, b *CgoCiphertext) (*CgoCiphertext, error) {
	var ptr *C.LuxFheCiphertext
	err := C.lux_fhe_xor(c.ptr, bsk.ptr, a.ptr, b.ptr, &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	result := &CgoCiphertext{ptr: ptr}
	runtime.SetFinalizer(result, (*CgoCiphertext).Free)
	return result, nil
}

// Nand evaluates NAND gate with bootstrapping
func (c *CgoContext) Nand(bsk *CgoBootstrapKey, a, b *CgoCiphertext) (*CgoCiphertext, error) {
	var ptr *C.LuxFheCiphertext
	err := C.lux_fhe_nand(c.ptr, bsk.ptr, a.ptr, b.ptr, &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	result := &CgoCiphertext{ptr: ptr}
	runtime.SetFinalizer(result, (*CgoCiphertext).Free)
	return result, nil
}

// Nor evaluates NOR gate with bootstrapping
func (c *CgoContext) Nor(bsk *CgoBootstrapKey, a, b *CgoCiphertext) (*CgoCiphertext, error) {
	var ptr *C.LuxFheCiphertext
	err := C.lux_fhe_nor(c.ptr, bsk.ptr, a.ptr, b.ptr, &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	result := &CgoCiphertext{ptr: ptr}
	runtime.SetFinalizer(result, (*CgoCiphertext).Free)
	return result, nil
}

// Xnor evaluates XNOR gate with bootstrapping
func (c *CgoContext) Xnor(bsk *CgoBootstrapKey, a, b *CgoCiphertext) (*CgoCiphertext, error) {
	var ptr *C.LuxFheCiphertext
	err := C.lux_fhe_xnor(c.ptr, bsk.ptr, a.ptr, b.ptr, &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	result := &CgoCiphertext{ptr: ptr}
	runtime.SetFinalizer(result, (*CgoCiphertext).Free)
	return result, nil
}

// Mux evaluates MUX gate (if sel then a else b)
func (c *CgoContext) Mux(bsk *CgoBootstrapKey, sel, a, b *CgoCiphertext) (*CgoCiphertext, error) {
	var ptr *C.LuxFheCiphertext
	err := C.lux_fhe_mux(c.ptr, bsk.ptr, sel.ptr, a.ptr, b.ptr, &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	result := &CgoCiphertext{ptr: ptr}
	runtime.SetFinalizer(result, (*CgoCiphertext).Free)
	return result, nil
}

// Bootstrap refreshes ciphertext to reduce noise
func (c *CgoContext) Bootstrap(bsk *CgoBootstrapKey, ct *CgoCiphertext) (*CgoCiphertext, error) {
	var ptr *C.LuxFheCiphertext
	err := C.lux_fhe_bootstrap(c.ptr, bsk.ptr, ct.ptr, &ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	result := &CgoCiphertext{ptr: ptr}
	runtime.SetFinalizer(result, (*CgoCiphertext).Free)
	return result, nil
}

// =============================================================================
// Serialization
// =============================================================================

// Marshal serializes ciphertext to bytes
func (ct *CgoCiphertext) Marshal() ([]byte, error) {
	var data *C.uint8_t
	var length C.size_t
	err := C.lux_fhe_ciphertext_marshal(ct.ptr, &data, &length)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	defer C.lux_fhe_bytes_free(data)
	return C.GoBytes(unsafe.Pointer(data), C.int(length)), nil
}

// UnmarshalCiphertext deserializes ciphertext from bytes
func (c *CgoContext) UnmarshalCiphertext(data []byte) (*CgoCiphertext, error) {
	var ptr *C.LuxFheCiphertext
	err := C.lux_fhe_ciphertext_unmarshal(c.ptr,
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
		&ptr)
	if err != C.LUX_FHE_OK {
		return nil, cgoErr(err)
	}
	ct := &CgoCiphertext{ptr: ptr}
	runtime.SetFinalizer(ct, (*CgoCiphertext).Free)
	return ct, nil
}

// =============================================================================
// Utility
// =============================================================================

// Version returns the C++ library version
func Version() string {
	return C.GoString(C.lux_fhe_version())
}

// HasGPU returns true if GPU acceleration is available
func HasGPU() bool {
	return bool(C.lux_fhe_has_gpu())
}

// IsCgoEnabled returns true (this file is only compiled with cgo)
func IsCgoEnabled() bool {
	return true
}
