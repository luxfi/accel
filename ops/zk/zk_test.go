package zk

import (
	"testing"
)

// Use smaller modulus for simpler tests
// 97 is prime and 97-1=96, 8|96, so 8th roots of unity exist
// A generator of Z/97* is 5
// The primitive 8th root of unity is 5^(96/8) = 5^12 mod 97 = 64
var smallNTTParams = NTTParams{
	N:       8,
	Modulus: 97, // Prime with 8 | (97-1)
	Root:    64, // 5^12 mod 97 = 64 is a primitive 8th root of unity
}

var smallFieldParams = FieldParams{
	Modulus: 97,
}

// Goldilocks prime for more realistic tests (fits in uint64)
// 2^64 - 2^32 + 1 has order 2^32, so 8th roots exist
var goldilocksNTTParams = NTTParams{
	N:       8,
	Modulus: 0xFFFFFFFF00000001, // 2^64 - 2^32 + 1 (Goldilocks)
	Root:    7,                  // Generator
}

var goldilocksFieldParams = FieldParams{
	Modulus: 0xFFFFFFFF00000001,
}

func TestNTTRoundtrip(t *testing.T) {
	coeffs := []uint64{1, 2, 3, 4, 5, 6, 7, 8}

	// Forward NTT
	evals, err := NTT(smallNTTParams, coeffs)
	if err != nil {
		t.Fatalf("NTT failed: %v", err)
	}

	// Inverse NTT
	recovered, err := INTT(smallNTTParams, evals)
	if err != nil {
		t.Fatalf("INTT failed: %v", err)
	}

	// Check roundtrip
	for i := range coeffs {
		if recovered[i] != coeffs[i] {
			t.Errorf("NTT roundtrip failed at index %d: got %d, want %d", i, recovered[i], coeffs[i])
		}
	}
}

func TestBatchNTT(t *testing.T) {
	polys := [][]uint64{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{8, 7, 6, 5, 4, 3, 2, 1},
	}

	// Forward batch NTT
	evals, err := BatchNTT(smallNTTParams, polys)
	if err != nil {
		t.Fatalf("BatchNTT failed: %v", err)
	}

	// Inverse batch NTT
	recovered, err := BatchINTT(smallNTTParams, evals)
	if err != nil {
		t.Fatalf("BatchINTT failed: %v", err)
	}

	// Check roundtrip
	for i := range polys {
		for j := range polys[i] {
			if recovered[i][j] != polys[i][j] {
				t.Errorf("BatchNTT roundtrip failed at [%d][%d]: got %d, want %d",
					i, j, recovered[i][j], polys[i][j])
			}
		}
	}
}

func TestPolyAdd(t *testing.T) {
	a := []uint64{1, 2, 3, 4}
	b := []uint64{5, 6, 7, 8}

	result, err := PolyAdd(smallFieldParams, a, b)
	if err != nil {
		t.Fatalf("PolyAdd failed: %v", err)
	}

	expected := []uint64{6, 8, 10, 12}
	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("PolyAdd[%d]: got %d, want %d", i, result[i], expected[i])
		}
	}
}

func TestPolySub(t *testing.T) {
	a := []uint64{10, 12, 14, 16}
	b := []uint64{5, 6, 7, 8}

	result, err := PolySub(smallFieldParams, a, b)
	if err != nil {
		t.Fatalf("PolySub failed: %v", err)
	}

	expected := []uint64{5, 6, 7, 8}
	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("PolySub[%d]: got %d, want %d", i, result[i], expected[i])
		}
	}
}

func TestPolyEval(t *testing.T) {
	// p(x) = 1 + 2x + 3x^2
	coeffs := []uint64{1, 2, 3}
	points := []uint64{0, 1, 2}

	result, err := PolyEval(smallFieldParams, coeffs, points)
	if err != nil {
		t.Fatalf("PolyEval failed: %v", err)
	}

	// p(0) = 1
	// p(1) = 1 + 2 + 3 = 6
	// p(2) = 1 + 4 + 12 = 17 mod 97 = 17
	expected := []uint64{1, 6, 17}
	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("PolyEval at x=%d: got %d, want %d", points[i], result[i], expected[i])
		}
	}
}

func TestFieldOps(t *testing.T) {
	a := []uint64{5, 10, 15}
	b := []uint64{3, 4, 5}

	// Test add
	sum, err := FieldAdd(smallFieldParams, a, b)
	if err != nil {
		t.Fatalf("FieldAdd failed: %v", err)
	}
	expectedSum := []uint64{8, 14, 20} // All < 97
	for i := range expectedSum {
		if sum[i] != expectedSum[i] {
			t.Errorf("FieldAdd[%d]: got %d, want %d", i, sum[i], expectedSum[i])
		}
	}

	// Test mul
	prod, err := FieldMul(smallFieldParams, a, b)
	if err != nil {
		t.Fatalf("FieldMul failed: %v", err)
	}
	// 5*3=15, 10*4=40, 15*5=75 (all < 97)
	expectedProd := []uint64{15, 40, 75}
	for i := range expectedProd {
		if prod[i] != expectedProd[i] {
			t.Errorf("FieldMul[%d]: got %d, want %d", i, prod[i], expectedProd[i])
		}
	}

	// Test inv
	c := []uint64{2, 3, 5}
	inv, err := FieldInv(smallFieldParams, c)
	if err != nil {
		t.Fatalf("FieldInv failed: %v", err)
	}
	// Check a * inv(a) = 1
	for i := range c {
		product := (c[i] * inv[i]) % 97
		if product != 1 {
			t.Errorf("FieldInv[%d]: %d * %d = %d, want 1", i, c[i], inv[i], product)
		}
	}
}

func TestFieldExp(t *testing.T) {
	a := []uint64{2, 3, 5}

	result, err := FieldExp(smallFieldParams, a, 4)
	if err != nil {
		t.Fatalf("FieldExp failed: %v", err)
	}

	// 2^4=16, 3^4=81, 5^4=625=43 (mod 97)
	expected := []uint64{16, 81, 43}
	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("FieldExp[%d]: got %d, want %d", i, result[i], expected[i])
		}
	}
}

func TestFRIFold(t *testing.T) {
	params := FRIParams{
		Modulus:      97,
		FoldFactor:   2,
		BlowupFactor: 2,
	}

	// Simple test: even length evaluations
	evals := []uint64{1, 2, 3, 4, 5, 6, 7, 8}
	alpha := uint64(3)

	folded, err := FRIFold(params, evals, alpha)
	if err != nil {
		t.Fatalf("FRIFold failed: %v", err)
	}

	if len(folded) != len(evals)/2 {
		t.Errorf("FRIFold: expected length %d, got %d", len(evals)/2, len(folded))
	}
}

func TestMSM(t *testing.T) {
	// Simple test - just verify it doesn't error
	scalars := [][]byte{{1, 0, 0, 0}, {2, 0, 0, 0}}
	points := [][]byte{make([]byte, 64), make([]byte, 64)}

	result, err := MSM(CurveBN254, scalars, points)
	if err != nil {
		t.Fatalf("MSM failed: %v", err)
	}

	if len(result) != 64 {
		t.Errorf("MSM: expected 64 byte result, got %d", len(result))
	}
}

func TestPoseidon2(t *testing.T) {
	// Simple test with minimal params
	params := Poseidon2Params{
		T:       3,
		D:       5,
		RoundsF: 4,
		RoundsP: 4,
		Modulus: 97,
	}

	inputs := []uint64{1, 2}

	_, err := Poseidon2(params, inputs)
	if err != nil {
		t.Fatalf("Poseidon2 failed: %v", err)
	}
}

func TestErrors(t *testing.T) {
	// Test invalid degree
	_, err := NTT(smallNTTParams, []uint64{1, 2, 3}) // Wrong length
	if err != ErrInvalidDegree {
		t.Errorf("Expected ErrInvalidDegree, got %v", err)
	}

	// Test dimension mismatch
	_, err = PolyAdd(smallFieldParams, []uint64{1, 2}, []uint64{1, 2, 3})
	if err != ErrDimensionMismatch {
		t.Errorf("Expected ErrDimensionMismatch, got %v", err)
	}

	// Test empty batch
	_, err = BatchNTT(smallNTTParams, [][]uint64{})
	if err != ErrEmptyBatch {
		t.Errorf("Expected ErrEmptyBatch, got %v", err)
	}
}

// Benchmarks

func BenchmarkNTT_N8(b *testing.B) {
	coeffs := []uint64{1, 2, 3, 4, 5, 6, 7, 8}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = NTT(smallNTTParams, coeffs)
	}
}

func BenchmarkPolyMul_N8(b *testing.B) {
	a := []uint64{1, 2, 3, 4, 5, 6, 7, 8}
	bc := []uint64{8, 7, 6, 5, 4, 3, 2, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = PolyMul(smallNTTParams, a, bc)
	}
}

func BenchmarkFieldMul(b *testing.B) {
	a := make([]uint64, 1024)
	bc := make([]uint64, 1024)
	for i := range a {
		a[i] = uint64(i + 1)
		bc[i] = uint64(i + 2)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = FieldMul(smallFieldParams, a, bc)
	}
}
