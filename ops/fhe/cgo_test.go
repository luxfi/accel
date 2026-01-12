//go:build cgo && luxfhe

package fhe

import "testing"

func TestCgoVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Fatal("version is empty")
	}
	t.Logf("luxfhe version: %s", v)
}

func TestCgoHasGPU(t *testing.T) {
	has := HasGPU()
	t.Logf("luxfhe has GPU: %v", has)
}

func TestCgoContextCreate(t *testing.T) {
	ctx, err := NewCgoContext(CgoParamsToy, CgoMethodLMKCDEY)
	if err != nil {
		t.Fatalf("failed to create context: %v", err)
	}
	defer ctx.Free()

	n := ctx.N()
	ringDim := ctx.RingDim()
	t.Logf("context N=%d, RingDim=%d", n, ringDim)
}
