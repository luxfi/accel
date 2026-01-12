//go:build accel

package fhe

import (
	"testing"
)

func TestMultiGPUInit(t *testing.T) {
	ctx, err := NewMultiGPUContext()
	if err != nil {
		t.Skipf("Multi-GPU initialization failed (no GPUs available?): %v", err)
		return
	}
	defer ctx.Close()

	gpuCount := ctx.GPUCount()
	if gpuCount <= 0 {
		t.Skipf("No GPUs available, skipping multi-GPU tests")
		return
	}

	t.Logf("Found %d GPU(s)", gpuCount)
	t.Logf("NVLink available: %v", ctx.HasNVLink())
}

func TestMultiGPUMemoryInfo(t *testing.T) {
	ctx, err := NewMultiGPUContext()
	if err != nil {
		t.Skipf("Multi-GPU initialization failed: %v", err)
		return
	}
	defer ctx.Close()

	if ctx.GPUCount() == 0 {
		t.Skip("No GPUs available")
		return
	}

	for i := 0; i < ctx.GPUCount(); i++ {
		memInfo, err := ctx.GetMemoryInfo(i)
		if err != nil {
			t.Errorf("Failed to get memory info for GPU %d: %v", i, err)
			continue
		}

		t.Logf("GPU %d: Free=%d MB, Total=%d MB",
			i,
			memInfo.FreeMem/(1024*1024),
			memInfo.TotalMem/(1024*1024))

		if memInfo.TotalMem <= 0 {
			t.Errorf("GPU %d has invalid total memory: %d", i, memInfo.TotalMem)
		}
	}
}

func TestMultiGPUWorkDistribution(t *testing.T) {
	ctx, err := NewMultiGPUContext()
	if err != nil {
		t.Skipf("Multi-GPU initialization failed: %v", err)
		return
	}
	defer ctx.Close()

	if ctx.GPUCount() == 0 {
		t.Skip("No GPUs available")
		return
	}

	totalElements := uint32(1024 * 1024)
	dist, err := ctx.DistributeWork(totalElements)
	if err != nil {
		t.Fatalf("Work distribution failed: %v", err)
	}

	var sum uint32
	for i, work := range dist.WorkPerGPU {
		t.Logf("GPU %d: offset=%d, work=%d", i, dist.Offsets[i], work)
		sum += work
	}

	if sum != totalElements {
		t.Errorf("Work distribution sum mismatch: expected %d, got %d", totalElements, sum)
	}
}

func TestMultiGPUBarrier(t *testing.T) {
	ctx, err := NewMultiGPUContext()
	if err != nil {
		t.Skipf("Multi-GPU initialization failed: %v", err)
		return
	}
	defer ctx.Close()

	if ctx.GPUCount() == 0 {
		t.Skip("No GPUs available")
		return
	}

	// Test that barrier doesn't crash
	ctx.Barrier()

	err = ctx.SyncAll()
	if err != nil {
		t.Errorf("SyncAll failed: %v", err)
	}
}

func TestMultiGPUClose(t *testing.T) {
	ctx, err := NewMultiGPUContext()
	if err != nil {
		t.Skipf("Multi-GPU initialization failed: %v", err)
		return
	}

	ctx.Close()

	// Operations after close should fail gracefully
	if count := ctx.GPUCount(); count != 0 {
		t.Errorf("Expected GPUCount=0 after close, got %d", count)
	}

	// Double close should not crash
	ctx.Close()
}

func TestSetDevice(t *testing.T) {
	ctx, err := NewMultiGPUContext()
	if err != nil {
		t.Skipf("Multi-GPU initialization failed: %v", err)
		return
	}
	defer ctx.Close()

	if ctx.GPUCount() == 0 {
		t.Skip("No GPUs available")
		return
	}

	// Set to first device
	err = SetDevice(0)
	if err != nil {
		t.Errorf("Failed to set device 0: %v", err)
	}

	// Try invalid device
	err = SetDevice(9999)
	if err == nil {
		t.Errorf("Expected error when setting invalid device, got nil")
	}
}

func TestDistributedBatchEncrypt(t *testing.T) {
	ctx, err := NewMultiGPUContext()
	if err != nil {
		t.Skipf("Multi-GPU initialization failed: %v", err)
		return
	}
	defer ctx.Close()

	// Test parameters
	params := Params{
		Scheme:     SchemeBFV,
		PolyDegree: 4096,
		CoeffMods:  []uint64{0xffffee001, 0xffffc4001},
		PlainMod:   65537,
	}

	// Generate keys
	sk, pk, err := KeyGen(params)
	if err != nil {
		t.Fatalf("KeyGen failed: %v", err)
	}

	// Create test plaintexts
	numPlaintexts := 10
	pts := make([]*Plaintext, numPlaintexts)
	for i := range pts {
		vals := make([]int64, 100)
		for j := range vals {
			vals[j] = int64(i*100 + j)
		}
		pt, err := EncodeBFV(params, vals)
		if err != nil {
			t.Fatalf("EncodeBFV failed: %v", err)
		}
		pts[i] = pt
	}

	// Distributed batch encrypt
	cts, err := DistributedBatchEncrypt(ctx, params, pk, pts)
	if err != nil {
		t.Fatalf("DistributedBatchEncrypt failed: %v", err)
	}

	if len(cts) != numPlaintexts {
		t.Errorf("Expected %d ciphertexts, got %d", numPlaintexts, len(cts))
	}

	// Verify by decrypting
	for i, ct := range cts {
		pt, err := Decrypt(params, sk, ct)
		if err != nil {
			t.Errorf("Decrypt %d failed: %v", i, err)
			continue
		}

		vals, err := DecodeBFV(params, pt)
		if err != nil {
			t.Errorf("DecodeBFV %d failed: %v", i, err)
			continue
		}

		// Check first few values
		for j := 0; j < 10 && j < len(vals); j++ {
			expected := int64(i*100 + j)
			if vals[j] != expected {
				t.Errorf("Value mismatch at ct[%d][%d]: expected %d, got %d", i, j, expected, vals[j])
			}
		}
	}
}
