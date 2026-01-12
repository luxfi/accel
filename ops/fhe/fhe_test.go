package fhe

import (
	"testing"
)

// Test parameters for BFV scheme
var testBFVParams = Params{
	Scheme:     SchemeBFV,
	PolyDegree: 4096,
	CoeffMods:  []uint64{0xFFFFFFFFFFFFFFFF - 58, 0xFFFFFFFFFFFFFFFF - 16}, // Simplified
	PlainMod:   65537,
	Scale:      1.0,
}

// Test parameters for CKKS scheme
var testCKKSParams = Params{
	Scheme:     SchemeCKKS,
	PolyDegree: 4096,
	CoeffMods:  []uint64{0xFFFFFFFFFFFFFFFF - 58, 0xFFFFFFFFFFFFFFFF - 16},
	PlainMod:   0,
	Scale:      1 << 40, // 2^40
}

func TestValidateParams(t *testing.T) {
	tests := []struct {
		name    string
		params  Params
		wantErr bool
	}{
		{
			name:    "valid BFV",
			params:  testBFVParams,
			wantErr: false,
		},
		{
			name:    "valid CKKS",
			params:  testCKKSParams,
			wantErr: false,
		},
		{
			name: "invalid degree (not power of 2)",
			params: Params{
				Scheme:     SchemeBFV,
				PolyDegree: 4095,
				CoeffMods:  []uint64{0xFFFFFFFFFFFFFFFF - 58},
				PlainMod:   65537,
			},
			wantErr: true,
		},
		{
			name: "BFV missing plaintext modulus",
			params: Params{
				Scheme:     SchemeBFV,
				PolyDegree: 4096,
				CoeffMods:  []uint64{0xFFFFFFFFFFFFFFFF - 58},
				PlainMod:   0,
			},
			wantErr: true,
		},
		{
			name: "CKKS missing scale",
			params: Params{
				Scheme:     SchemeCKKS,
				PolyDegree: 4096,
				CoeffMods:  []uint64{0xFFFFFFFFFFFFFFFF - 58},
				Scale:      0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateParams(tt.params)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateParams() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestKeyGen(t *testing.T) {
	sk, pk, err := KeyGen(testBFVParams)
	if err != nil {
		t.Fatalf("KeyGen() error = %v", err)
	}
	if sk == nil || len(sk.Data) == 0 {
		t.Error("KeyGen() returned nil or empty secret key")
	}
	if pk == nil || len(pk.Data) == 0 {
		t.Error("KeyGen() returned nil or empty public key")
	}
}

func TestGenRelinKey(t *testing.T) {
	sk, _, err := KeyGen(testBFVParams)
	if err != nil {
		t.Fatalf("KeyGen() error = %v", err)
	}

	rlk, err := GenRelinKey(testBFVParams, sk)
	if err != nil {
		t.Fatalf("GenRelinKey() error = %v", err)
	}
	if rlk == nil || len(rlk.Data) == 0 {
		t.Error("GenRelinKey() returned nil or empty key")
	}
}

func TestGenGaloisKey(t *testing.T) {
	sk, _, err := KeyGen(testBFVParams)
	if err != nil {
		t.Fatalf("KeyGen() error = %v", err)
	}

	steps := []int{1, 2, 4, 8}
	gk, err := GenGaloisKey(testBFVParams, sk, steps)
	if err != nil {
		t.Fatalf("GenGaloisKey() error = %v", err)
	}
	if gk == nil || len(gk.Data) == 0 {
		t.Error("GenGaloisKey() returned nil or empty key")
	}
	if len(gk.Steps) != len(steps) {
		t.Errorf("GenGaloisKey() steps = %v, want %v", gk.Steps, steps)
	}
}

func TestBFVEncryptDecrypt(t *testing.T) {
	sk, pk, err := KeyGen(testBFVParams)
	if err != nil {
		t.Fatalf("KeyGen() error = %v", err)
	}

	// Encode integers
	values := []int64{1, 2, 3, 4, 5}
	pt, err := EncodeBFV(testBFVParams, values)
	if err != nil {
		t.Fatalf("EncodeBFV() error = %v", err)
	}

	// Encrypt
	ct, err := Encrypt(testBFVParams, pk, pt)
	if err != nil {
		t.Fatalf("Encrypt() error = %v", err)
	}
	if ct.Scheme != SchemeBFV {
		t.Errorf("Encrypt() scheme = %v, want BFV", ct.Scheme)
	}

	// Decrypt
	ptDec, err := Decrypt(testBFVParams, sk, ct)
	if err != nil {
		t.Fatalf("Decrypt() error = %v", err)
	}
	if ptDec.Scheme != SchemeBFV {
		t.Errorf("Decrypt() scheme = %v, want BFV", ptDec.Scheme)
	}
}

func TestCKKSEncryptDecrypt(t *testing.T) {
	sk, pk, err := KeyGen(testCKKSParams)
	if err != nil {
		t.Fatalf("KeyGen() error = %v", err)
	}

	// Encode floats
	values := []float64{1.0, 2.5, 3.14, 4.0, 5.5}
	pt, err := EncodeCKKS(testCKKSParams, values)
	if err != nil {
		t.Fatalf("EncodeCKKS() error = %v", err)
	}

	// Encrypt
	ct, err := Encrypt(testCKKSParams, pk, pt)
	if err != nil {
		t.Fatalf("Encrypt() error = %v", err)
	}
	if ct.Scheme != SchemeCKKS {
		t.Errorf("Encrypt() scheme = %v, want CKKS", ct.Scheme)
	}

	// Decrypt
	ptDec, err := Decrypt(testCKKSParams, sk, ct)
	if err != nil {
		t.Fatalf("Decrypt() error = %v", err)
	}
	if ptDec.Scheme != SchemeCKKS {
		t.Errorf("Decrypt() scheme = %v, want CKKS", ptDec.Scheme)
	}
}

func TestAdd(t *testing.T) {
	_, pk, err := KeyGen(testBFVParams)
	if err != nil {
		t.Fatalf("KeyGen() error = %v", err)
	}

	pt1, _ := EncodeBFV(testBFVParams, []int64{1, 2, 3})
	pt2, _ := EncodeBFV(testBFVParams, []int64{4, 5, 6})

	ct1, _ := Encrypt(testBFVParams, pk, pt1)
	ct2, _ := Encrypt(testBFVParams, pk, pt2)

	result, err := Add(testBFVParams, ct1, ct2)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if result == nil {
		t.Error("Add() returned nil")
	}
}

func TestMultiply(t *testing.T) {
	sk, pk, err := KeyGen(testBFVParams)
	if err != nil {
		t.Fatalf("KeyGen() error = %v", err)
	}

	rlk, err := GenRelinKey(testBFVParams, sk)
	if err != nil {
		t.Fatalf("GenRelinKey() error = %v", err)
	}

	pt1, _ := EncodeBFV(testBFVParams, []int64{2, 3})
	pt2, _ := EncodeBFV(testBFVParams, []int64{4, 5})

	ct1, _ := Encrypt(testBFVParams, pk, pt1)
	ct2, _ := Encrypt(testBFVParams, pk, pt2)

	result, err := Multiply(testBFVParams, ct1, ct2, rlk)
	if err != nil {
		t.Fatalf("Multiply() error = %v", err)
	}
	if result == nil {
		t.Error("Multiply() returned nil")
	}
}

func TestRotate(t *testing.T) {
	sk, pk, err := KeyGen(testBFVParams)
	if err != nil {
		t.Fatalf("KeyGen() error = %v", err)
	}

	steps := []int{1, -1, 2}
	gk, err := GenGaloisKey(testBFVParams, sk, steps)
	if err != nil {
		t.Fatalf("GenGaloisKey() error = %v", err)
	}

	pt, _ := EncodeBFV(testBFVParams, []int64{1, 2, 3, 4})
	ct, _ := Encrypt(testBFVParams, pk, pt)

	result, err := Rotate(testBFVParams, ct, gk, 1)
	if err != nil {
		t.Fatalf("Rotate() error = %v", err)
	}
	if result == nil {
		t.Error("Rotate() returned nil")
	}
}

func TestRotateInvalidStep(t *testing.T) {
	sk, pk, err := KeyGen(testBFVParams)
	if err != nil {
		t.Fatalf("KeyGen() error = %v", err)
	}

	steps := []int{1, 2}
	gk, _ := GenGaloisKey(testBFVParams, sk, steps)
	pt, _ := EncodeBFV(testBFVParams, []int64{1, 2, 3, 4})
	ct, _ := Encrypt(testBFVParams, pk, pt)

	_, err = Rotate(testBFVParams, ct, gk, 5) // 5 not in steps
	if err != ErrRotationKey {
		t.Errorf("Rotate() error = %v, want ErrRotationKey", err)
	}
}

func TestRescale(t *testing.T) {
	_, pk, err := KeyGen(testCKKSParams)
	if err != nil {
		t.Fatalf("KeyGen() error = %v", err)
	}

	pt, _ := EncodeCKKS(testCKKSParams, []float64{1.0, 2.0})
	ct, _ := Encrypt(testCKKSParams, pk, pt)

	result, err := Rescale(testCKKSParams, ct)
	if err != nil {
		t.Fatalf("Rescale() error = %v", err)
	}
	if result.Level != ct.Level-1 {
		t.Errorf("Rescale() level = %d, want %d", result.Level, ct.Level-1)
	}
}

func TestBatchEncrypt(t *testing.T) {
	_, pk, err := KeyGen(testBFVParams)
	if err != nil {
		t.Fatalf("KeyGen() error = %v", err)
	}

	pts := make([]*Plaintext, 10)
	for i := range pts {
		pts[i], _ = EncodeBFV(testBFVParams, []int64{int64(i)})
	}

	cts, err := BatchEncrypt(testBFVParams, pk, pts)
	if err != nil {
		t.Fatalf("BatchEncrypt() error = %v", err)
	}
	if len(cts) != len(pts) {
		t.Errorf("BatchEncrypt() returned %d ciphertexts, want %d", len(cts), len(pts))
	}
}

func TestInputValidation(t *testing.T) {
	_, err := Encrypt(testBFVParams, nil, nil)
	if err != ErrInvalidInput {
		t.Errorf("Encrypt(nil, nil) error = %v, want ErrInvalidInput", err)
	}

	_, err = Decrypt(testBFVParams, nil, nil)
	if err != ErrInvalidInput {
		t.Errorf("Decrypt(nil, nil) error = %v, want ErrInvalidInput", err)
	}

	_, err = Add(testBFVParams, nil, nil)
	if err != ErrInvalidInput {
		t.Errorf("Add(nil, nil) error = %v, want ErrInvalidInput", err)
	}

	_, err = GenRelinKey(testBFVParams, nil)
	if err != ErrInvalidInput {
		t.Errorf("GenRelinKey(nil) error = %v, want ErrInvalidInput", err)
	}

	_, err = GenGaloisKey(testBFVParams, nil, nil)
	if err != ErrInvalidInput {
		t.Errorf("GenGaloisKey(nil, nil) error = %v, want ErrInvalidInput", err)
	}
}

func TestHasStep(t *testing.T) {
	steps := []int{1, 2, 4, 8}

	if !hasStep(steps, 1) {
		t.Error("hasStep(steps, 1) = false, want true")
	}
	if !hasStep(steps, 4) {
		t.Error("hasStep(steps, 4) = false, want true")
	}
	if hasStep(steps, 3) {
		t.Error("hasStep(steps, 3) = true, want false")
	}
	if hasStep(steps, 16) {
		t.Error("hasStep(steps, 16) = true, want false")
	}
}

func BenchmarkBFVEncrypt(b *testing.B) {
	_, pk, _ := KeyGen(testBFVParams)
	pt, _ := EncodeBFV(testBFVParams, []int64{1, 2, 3, 4, 5})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Encrypt(testBFVParams, pk, pt)
	}
}

func BenchmarkBFVMultiply(b *testing.B) {
	sk, pk, _ := KeyGen(testBFVParams)
	rlk, _ := GenRelinKey(testBFVParams, sk)
	pt, _ := EncodeBFV(testBFVParams, []int64{1, 2, 3, 4, 5})
	ct1, _ := Encrypt(testBFVParams, pk, pt)
	ct2, _ := Encrypt(testBFVParams, pk, pt)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Multiply(testBFVParams, ct1, ct2, rlk)
	}
}

func BenchmarkCKKSEncrypt(b *testing.B) {
	_, pk, _ := KeyGen(testCKKSParams)
	pt, _ := EncodeCKKS(testCKKSParams, []float64{1.0, 2.0, 3.0, 4.0, 5.0})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Encrypt(testCKKSParams, pk, pt)
	}
}
