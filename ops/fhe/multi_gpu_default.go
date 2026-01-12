//go:build !accel

// Package fhe provides Fully Homomorphic Encryption operations.
// This is the pure Go fallback when accel build tag is not present.
//
// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
package fhe

import (
	"fmt"
	"sync"
	"unsafe"
)

// MultiGPUContext is a stub for multi-GPU coordination when accel is not available.
// All operations fall back to single-threaded CPU execution.
type MultiGPUContext struct {
	mu sync.RWMutex
}

// NewMultiGPUContext returns a stub context when accel is not available.
func NewMultiGPUContext() (*MultiGPUContext, error) {
	return &MultiGPUContext{}, nil
}

// Close is a no-op for the stub context.
func (ctx *MultiGPUContext) Close() {}

// GPUCount returns 0 when accel is not available.
func (ctx *MultiGPUContext) GPUCount() int {
	return 0
}

// HasNVLink returns false when accel is not available.
func (ctx *MultiGPUContext) HasNVLink() bool {
	return false
}

// P2PCopy returns an error when accel is not available.
func (ctx *MultiGPUContext) P2PCopy(srcDevice int, srcPtr unsafe.Pointer, dstDevice int, dstPtr unsafe.Pointer, size int) error {
	return fmt.Errorf("multi-GPU not available: accel build tag not enabled")
}

// Barrier is a no-op when accel is not available.
func (ctx *MultiGPUContext) Barrier() {}

// WorkDistribution describes how work is distributed across GPUs.
type WorkDistribution struct {
	WorkPerGPU []uint32
	Offsets    []uint32
}

// DistributeWork returns all work for a single CPU when accel is not available.
func (ctx *MultiGPUContext) DistributeWork(totalElements uint32) (*WorkDistribution, error) {
	return &WorkDistribution{
		WorkPerGPU: []uint32{totalElements},
		Offsets:    []uint32{0},
	}, nil
}

// DistributedNTTData holds data for NTT operations.
type DistributedNTTData struct {
	DataPerGPU     []unsafe.Pointer
	TwiddlesPerGPU []unsafe.Pointer
	NPerGPU        []uint32
	Q              uint64
}

// NTTForward returns an error when accel is not available.
func (ctx *MultiGPUContext) NTTForward(data *DistributedNTTData) error {
	return fmt.Errorf("distributed NTT not available: accel build tag not enabled")
}

// NTTInverse returns an error when accel is not available.
func (ctx *MultiGPUContext) NTTInverse(data *DistributedNTTData) error {
	return fmt.Errorf("distributed NTT not available: accel build tag not enabled")
}

// GPUMemoryInfo holds memory information for a GPU.
type GPUMemoryInfo struct {
	FreeMem  int64
	TotalMem int64
}

// GetMemoryInfo returns an error when accel is not available.
func (ctx *MultiGPUContext) GetMemoryInfo(device int) (*GPUMemoryInfo, error) {
	return nil, fmt.Errorf("GPU memory info not available: accel build tag not enabled")
}

// SetDevice returns an error when accel is not available.
func SetDevice(device int) error {
	return fmt.Errorf("set device not available: accel build tag not enabled")
}

// SyncDevice returns an error when accel is not available.
func (ctx *MultiGPUContext) SyncDevice(device int) error {
	return fmt.Errorf("sync device not available: accel build tag not enabled")
}

// SyncAll returns an error when accel is not available.
func (ctx *MultiGPUContext) SyncAll() error {
	return fmt.Errorf("sync all not available: accel build tag not enabled")
}

// GetStream returns nil when accel is not available.
func (ctx *MultiGPUContext) GetStream(device int) unsafe.Pointer {
	return nil
}

// DistributedBatchEncrypt falls back to single-threaded CPU encryption.
func DistributedBatchEncrypt(mgpu *MultiGPUContext, params Params, pk *PublicKey, pts []*Plaintext) ([]*Ciphertext, error) {
	if pk == nil || len(pts) == 0 {
		return nil, ErrInvalidInput
	}
	return batchEncryptCPU(params, pk, pts)
}

// DistributedBatchMultiply falls back to single-threaded CPU multiplication.
func DistributedBatchMultiply(mgpu *MultiGPUContext, params Params, ct1s, ct2s []*Ciphertext, rlk *RelinKey) ([]*Ciphertext, error) {
	if rlk == nil || len(ct1s) == 0 || len(ct1s) != len(ct2s) {
		return nil, ErrInvalidInput
	}

	results := make([]*Ciphertext, len(ct1s))
	for i := range ct1s {
		ct, err := multiplyCPU(params, ct1s[i], ct2s[i], rlk)
		if err != nil {
			return nil, err
		}
		results[i] = ct
	}
	return results, nil
}
