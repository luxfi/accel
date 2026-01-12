//go:build accel

// Package fhe provides GPU-accelerated Fully Homomorphic Encryption operations.
//
// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
package fhe

/*
#cgo CFLAGS: -I${SRCDIR}/../../../internal/capi
#cgo CXXFLAGS: -I${SRCDIR}/../../../internal/capi -std=c++17 -O3
#cgo linux LDFLAGS: -lcudart -lfhe_multi_gpu
#cgo darwin LDFLAGS: -framework Metal -framework Foundation -lfhe_multi_gpu

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

// Multi-GPU context handle
typedef struct MultiGPUContext MultiGPUContext;

// Multi-GPU C API declarations
int multi_gpu_init(MultiGPUContext** ctx);
void multi_gpu_cleanup(MultiGPUContext* ctx);
int multi_gpu_count(MultiGPUContext* ctx);
bool multi_gpu_has_nvlink(MultiGPUContext* ctx);
int multi_gpu_p2p_copy(MultiGPUContext* ctx, int src_dev, void* src, int dst_dev, void* dst, size_t size);
void multi_gpu_barrier(MultiGPUContext* ctx);
int multi_gpu_distribute_work(MultiGPUContext* ctx, uint32_t total, uint32_t* work_per_gpu, uint32_t* offsets);
int multi_gpu_ntt_forward(MultiGPUContext* ctx, uint64_t** data, uint64_t** twiddles, uint32_t* n_per_gpu, uint64_t q);
int multi_gpu_ntt_inverse(MultiGPUContext* ctx, uint64_t** data, uint64_t** twiddles, uint32_t* n_per_gpu, uint64_t q);
int multi_gpu_memory_info(MultiGPUContext* ctx, int device, size_t* free_mem, size_t* total_mem);
int multi_gpu_set_device(int device);
int multi_gpu_sync_device(MultiGPUContext* ctx, int device);
int multi_gpu_sync_all(MultiGPUContext* ctx);
void* multi_gpu_get_stream(MultiGPUContext* ctx, int device);
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/luxfi/accel"
)

// MultiGPUContext manages coordination across multiple GPUs for FHE operations.
type MultiGPUContext struct {
	ptr *C.MultiGPUContext
	mu  sync.RWMutex
}

// NewMultiGPUContext initializes a multi-GPU context.
// Returns error if no GPUs are available or initialization fails.
func NewMultiGPUContext() (*MultiGPUContext, error) {
	var ptr *C.MultiGPUContext
	ret := C.multi_gpu_init(&ptr)
	if ret != 0 || ptr == nil {
		return nil, fmt.Errorf("failed to initialize multi-GPU context: error code %d", ret)
	}

	return &MultiGPUContext{
		ptr: ptr,
	}, nil
}

// Close releases all GPU resources and cleans up the context.
func (ctx *MultiGPUContext) Close() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()

	if ctx.ptr != nil {
		C.multi_gpu_cleanup(ctx.ptr)
		ctx.ptr = nil
	}
}

// GPUCount returns the number of available GPUs.
func (ctx *MultiGPUContext) GPUCount() int {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.ptr == nil {
		return 0
	}

	return int(C.multi_gpu_count(ctx.ptr))
}

// HasNVLink returns true if NVLink is available between any GPUs.
// NVLink enables high-speed peer-to-peer transfers.
func (ctx *MultiGPUContext) HasNVLink() bool {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.ptr == nil {
		return false
	}

	return bool(C.multi_gpu_has_nvlink(ctx.ptr))
}

// P2PCopy performs peer-to-peer memory copy between GPUs.
// Uses NVLink when available, falls back to host staging otherwise.
// srcDevice and dstDevice are GPU indices [0, GPUCount).
func (ctx *MultiGPUContext) P2PCopy(srcDevice int, srcPtr unsafe.Pointer, dstDevice int, dstPtr unsafe.Pointer, size int) error {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.ptr == nil {
		return fmt.Errorf("multi-GPU context is closed")
	}

	ret := C.multi_gpu_p2p_copy(
		ctx.ptr,
		C.int(srcDevice),
		srcPtr,
		C.int(dstDevice),
		dstPtr,
		C.size_t(size),
	)

	if ret != 0 {
		return fmt.Errorf("P2P copy failed: error code %d", ret)
	}

	return nil
}

// Barrier synchronizes all GPUs using events.
// Blocks until all GPU streams have completed pending work.
func (ctx *MultiGPUContext) Barrier() {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.ptr == nil {
		return
	}

	C.multi_gpu_barrier(ctx.ptr)
}

// WorkDistribution describes how work is distributed across GPUs.
type WorkDistribution struct {
	WorkPerGPU []uint32 // Number of elements assigned to each GPU
	Offsets    []uint32 // Starting offset for each GPU's work
}

// DistributeWork calculates optimal work distribution across GPUs.
// Uses load balancing based on available GPU memory.
func (ctx *MultiGPUContext) DistributeWork(totalElements uint32) (*WorkDistribution, error) {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.ptr == nil {
		return nil, fmt.Errorf("multi-GPU context is closed")
	}

	numGPUs := int(C.multi_gpu_count(ctx.ptr))
	workPerGPU := make([]uint32, numGPUs)
	offsets := make([]uint32, numGPUs)

	ret := C.multi_gpu_distribute_work(
		ctx.ptr,
		C.uint32_t(totalElements),
		(*C.uint32_t)(unsafe.Pointer(&workPerGPU[0])),
		(*C.uint32_t)(unsafe.Pointer(&offsets[0])),
	)

	if ret != 0 {
		return nil, fmt.Errorf("work distribution failed: error code %d", ret)
	}

	return &WorkDistribution{
		WorkPerGPU: workPerGPU,
		Offsets:    offsets,
	}, nil
}

// DistributedNTTData holds data distributed across multiple GPUs for NTT operations.
type DistributedNTTData struct {
	DataPerGPU     []unsafe.Pointer // GPU memory pointers for data on each GPU
	TwiddlesPerGPU []unsafe.Pointer // GPU memory pointers for twiddle factors on each GPU
	NPerGPU        []uint32         // Number of elements on each GPU
	Q              uint64           // Modulus
}

// NTTForward performs distributed forward NTT across multiple GPUs.
// Uses four-step algorithm to decompose work across GPUs.
func (ctx *MultiGPUContext) NTTForward(data *DistributedNTTData) error {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.ptr == nil {
		return fmt.Errorf("multi-GPU context is closed")
	}

	if data == nil {
		return fmt.Errorf("distributed NTT data is nil")
	}

	numGPUs := int(C.multi_gpu_count(ctx.ptr))
	if len(data.DataPerGPU) != numGPUs || len(data.TwiddlesPerGPU) != numGPUs || len(data.NPerGPU) != numGPUs {
		return fmt.Errorf("data arrays size mismatch: expected %d GPUs", numGPUs)
	}

	// Convert Go slices to C arrays
	cDataPtrs := make([]*C.uint64_t, numGPUs)
	cTwiddlePtrs := make([]*C.uint64_t, numGPUs)
	for i := 0; i < numGPUs; i++ {
		cDataPtrs[i] = (*C.uint64_t)(data.DataPerGPU[i])
		cTwiddlePtrs[i] = (*C.uint64_t)(data.TwiddlesPerGPU[i])
	}

	ret := C.multi_gpu_ntt_forward(
		ctx.ptr,
		(**C.uint64_t)(unsafe.Pointer(&cDataPtrs[0])),
		(**C.uint64_t)(unsafe.Pointer(&cTwiddlePtrs[0])),
		(*C.uint32_t)(unsafe.Pointer(&data.NPerGPU[0])),
		C.uint64_t(data.Q),
	)

	if ret != 0 {
		return fmt.Errorf("distributed NTT forward failed: error code %d", ret)
	}

	return nil
}

// NTTInverse performs distributed inverse NTT across multiple GPUs.
func (ctx *MultiGPUContext) NTTInverse(data *DistributedNTTData) error {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.ptr == nil {
		return fmt.Errorf("multi-GPU context is closed")
	}

	if data == nil {
		return fmt.Errorf("distributed NTT data is nil")
	}

	numGPUs := int(C.multi_gpu_count(ctx.ptr))
	if len(data.DataPerGPU) != numGPUs || len(data.TwiddlesPerGPU) != numGPUs || len(data.NPerGPU) != numGPUs {
		return fmt.Errorf("data arrays size mismatch: expected %d GPUs", numGPUs)
	}

	// Convert Go slices to C arrays
	cDataPtrs := make([]*C.uint64_t, numGPUs)
	cTwiddlePtrs := make([]*C.uint64_t, numGPUs)
	for i := 0; i < numGPUs; i++ {
		cDataPtrs[i] = (*C.uint64_t)(data.DataPerGPU[i])
		cTwiddlePtrs[i] = (*C.uint64_t)(data.TwiddlesPerGPU[i])
	}

	ret := C.multi_gpu_ntt_inverse(
		ctx.ptr,
		(**C.uint64_t)(unsafe.Pointer(&cDataPtrs[0])),
		(**C.uint64_t)(unsafe.Pointer(&cTwiddlePtrs[0])),
		(*C.uint32_t)(unsafe.Pointer(&data.NPerGPU[0])),
		C.uint64_t(data.Q),
	)

	if ret != 0 {
		return fmt.Errorf("distributed NTT inverse failed: error code %d", ret)
	}

	return nil
}

// GPUMemoryInfo holds memory information for a GPU.
type GPUMemoryInfo struct {
	FreeMem  int64 // Free memory in bytes
	TotalMem int64 // Total memory in bytes
}

// GetMemoryInfo returns memory information for a specific GPU.
func (ctx *MultiGPUContext) GetMemoryInfo(device int) (*GPUMemoryInfo, error) {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.ptr == nil {
		return nil, fmt.Errorf("multi-GPU context is closed")
	}

	var freeMem, totalMem C.size_t
	ret := C.multi_gpu_memory_info(
		ctx.ptr,
		C.int(device),
		&freeMem,
		&totalMem,
	)

	if ret != 0 {
		return nil, fmt.Errorf("failed to get memory info for GPU %d: error code %d", device, ret)
	}

	return &GPUMemoryInfo{
		FreeMem:  int64(freeMem),
		TotalMem: int64(totalMem),
	}, nil
}

// SetDevice sets the active GPU device.
func SetDevice(device int) error {
	ret := C.multi_gpu_set_device(C.int(device))
	if ret != 0 {
		return fmt.Errorf("failed to set device %d: error code %d", device, ret)
	}
	return nil
}

// SyncDevice synchronizes a specific GPU stream.
func (ctx *MultiGPUContext) SyncDevice(device int) error {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.ptr == nil {
		return fmt.Errorf("multi-GPU context is closed")
	}

	ret := C.multi_gpu_sync_device(ctx.ptr, C.int(device))
	if ret != 0 {
		return fmt.Errorf("failed to sync device %d: error code %d", device, ret)
	}

	return nil
}

// SyncAll synchronizes all GPU streams.
func (ctx *MultiGPUContext) SyncAll() error {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.ptr == nil {
		return fmt.Errorf("multi-GPU context is closed")
	}

	ret := C.multi_gpu_sync_all(ctx.ptr)
	if ret != 0 {
		return fmt.Errorf("failed to sync all devices: error code %d", ret)
	}

	return nil
}

// GetStream returns the GPU stream for a specific device.
// Returns nil if device is invalid or context is closed.
func (ctx *MultiGPUContext) GetStream(device int) unsafe.Pointer {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.ptr == nil {
		return nil
	}

	stream := C.multi_gpu_get_stream(ctx.ptr, C.int(device))
	return unsafe.Pointer(stream)
}

// DistributedBatchEncrypt encrypts plaintexts distributed across multiple GPUs.
func DistributedBatchEncrypt(mgpu *MultiGPUContext, params Params, pk *PublicKey, pts []*Plaintext) ([]*Ciphertext, error) {
	if mgpu == nil || pk == nil || len(pts) == 0 {
		return nil, ErrInvalidInput
	}

	numGPUs := mgpu.GPUCount()
	if numGPUs <= 1 {
		// Fall back to single-GPU batch encrypt
		sess, err := accel.NewSession()
		if err != nil {
			return batchEncryptCPU(params, pk, pts)
		}
		defer sess.Close()
		return batchEncryptGPU(sess, params, pk, pts)
	}

	// Distribute work across GPUs
	dist, err := mgpu.DistributeWork(uint32(len(pts)))
	if err != nil {
		return nil, err
	}

	results := make([]*Ciphertext, len(pts))
	var wg sync.WaitGroup
	var encryptErr error
	var errMu sync.Mutex

	for gpuIdx := 0; gpuIdx < numGPUs; gpuIdx++ {
		offset := dist.Offsets[gpuIdx]
		count := dist.WorkPerGPU[gpuIdx]
		if count == 0 {
			continue
		}

		wg.Add(1)
		go func(gpu int, off, cnt uint32) {
			defer wg.Done()

			if err := SetDevice(gpu); err != nil {
				errMu.Lock()
				if encryptErr == nil {
					encryptErr = err
				}
				errMu.Unlock()
				return
			}

			// Encrypt batch for this GPU
			batch := pts[off : off+cnt]
			cts, err := batchEncryptCPU(params, pk, batch) // Use CPU for now
			if err != nil {
				errMu.Lock()
				if encryptErr == nil {
					encryptErr = err
				}
				errMu.Unlock()
				return
			}

			// Copy results
			for i, ct := range cts {
				results[int(off)+i] = ct
			}
		}(gpuIdx, offset, count)
	}

	wg.Wait()

	if encryptErr != nil {
		return nil, encryptErr
	}

	mgpu.Barrier()
	return results, nil
}

// DistributedBatchMultiply performs distributed ciphertext multiplication.
func DistributedBatchMultiply(mgpu *MultiGPUContext, params Params, ct1s, ct2s []*Ciphertext, rlk *RelinKey) ([]*Ciphertext, error) {
	if mgpu == nil || rlk == nil || len(ct1s) == 0 || len(ct1s) != len(ct2s) {
		return nil, ErrInvalidInput
	}

	numGPUs := mgpu.GPUCount()
	if numGPUs <= 1 {
		// Fall back to sequential
		results := make([]*Ciphertext, len(ct1s))
		for i := range ct1s {
			ct, err := Multiply(params, ct1s[i], ct2s[i], rlk)
			if err != nil {
				return nil, err
			}
			results[i] = ct
		}
		return results, nil
	}

	// Distribute work
	dist, err := mgpu.DistributeWork(uint32(len(ct1s)))
	if err != nil {
		return nil, err
	}

	results := make([]*Ciphertext, len(ct1s))
	var wg sync.WaitGroup
	var mulErr error
	var errMu sync.Mutex

	for gpuIdx := 0; gpuIdx < numGPUs; gpuIdx++ {
		offset := dist.Offsets[gpuIdx]
		count := dist.WorkPerGPU[gpuIdx]
		if count == 0 {
			continue
		}

		wg.Add(1)
		go func(gpu int, off, cnt uint32) {
			defer wg.Done()

			if err := SetDevice(gpu); err != nil {
				errMu.Lock()
				if mulErr == nil {
					mulErr = err
				}
				errMu.Unlock()
				return
			}

			// Process batch for this GPU
			for i := uint32(0); i < cnt; i++ {
				idx := int(off + i)
				ct, err := multiplyCPU(params, ct1s[idx], ct2s[idx], rlk)
				if err != nil {
					errMu.Lock()
					if mulErr == nil {
						mulErr = err
					}
					errMu.Unlock()
					return
				}
				results[idx] = ct
			}
		}(gpuIdx, offset, count)
	}

	wg.Wait()

	if mulErr != nil {
		return nil, mulErr
	}

	mgpu.Barrier()
	return results, nil
}
