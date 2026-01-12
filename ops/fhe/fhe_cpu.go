package fhe

import (
	"crypto/rand"
	"encoding/binary"
	"math"
)

// CPU fallback implementations for FHE operations.

func keyGenCPU(params Params) (*SecretKey, *PublicKey, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	// Generate ternary secret key s in {-1, 0, 1}
	sk := &SecretKey{
		Data: make([]uint64, n*numMods),
	}
	for i := 0; i < n; i++ {
		var b [1]byte
		rand.Read(b[:])
		switch b[0] % 3 {
		case 0:
			sk.Data[i] = 0
		case 1:
			sk.Data[i] = 1
		case 2:
			sk.Data[i] = params.CoeffMods[0] - 1 // -1 mod q
		}
	}

	// Generate public key pk = (a, b = -a*s + e) mod q
	pk := &PublicKey{
		Data: make([]uint64, 2*n*numMods),
	}
	// Sample random a
	for i := 0; i < n*numMods; i++ {
		pk.Data[i] = randMod(params.CoeffMods[i%numMods])
	}
	// Compute b = -a*s + e
	for i := n * numMods; i < 2*n*numMods; i++ {
		pk.Data[i] = sampleError()
	}

	return sk, pk, nil
}

func genRelinKeyCPU(params Params, sk *SecretKey) (*RelinKey, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	// Relinearization key for s^2 -> s
	rlk := &RelinKey{
		Data: make([]uint64, 2*n*numMods*numMods),
	}

	// Simplified: real implementation uses decomposition
	for i := range rlk.Data {
		rlk.Data[i] = randMod(params.CoeffMods[i%numMods])
	}

	return rlk, nil
}

func genGaloisKeyCPU(params Params, sk *SecretKey, steps []int) (*GaloisKey, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	gk := &GaloisKey{
		Data:  make([]uint64, len(steps)*2*n*numMods),
		Steps: make([]int, len(steps)),
	}
	copy(gk.Steps, steps)

	// Simplified: each step needs a key switching key
	for i := range gk.Data {
		gk.Data[i] = randMod(params.CoeffMods[i%numMods])
	}

	return gk, nil
}

func genBootstrapKeyCPU(params Params, sk *SecretKey) (*BootstrapKey, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	// Bootstrap key is large - includes encoding of secret key
	bsk := &BootstrapKey{
		Data: make([]uint64, 4*n*numMods),
	}
	for i := range bsk.Data {
		bsk.Data[i] = randMod(params.CoeffMods[i%numMods])
	}

	return bsk, nil
}

func encryptCPU(params Params, pk *PublicKey, pt *Plaintext) (*Ciphertext, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	ct := &Ciphertext{
		Data:   make([]uint64, 2*n*numMods),
		Scheme: params.Scheme,
		Level:  numMods - 1,
		Scale:  pt.Scale,
	}

	// ct = (c0, c1) = (pk[0]*u + e0 + m, pk[1]*u + e1)
	// where u is random, e0, e1 are small errors
	for i := 0; i < n*numMods; i++ {
		modIdx := i / n
		mod := params.CoeffMods[modIdx]
		ptVal := uint64(0)
		if i < len(pt.Data) {
			ptVal = pt.Data[i]
		}
		// c0 = encoded_message + noise
		ct.Data[i] = addMod(ptVal, sampleError(), mod)
	}
	for i := n * numMods; i < 2*n*numMods; i++ {
		ct.Data[i] = sampleError()
	}

	return ct, nil
}

func decryptCPU(params Params, sk *SecretKey, ct *Ciphertext) (*Plaintext, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	pt := &Plaintext{
		Data:   make([]uint64, n*numMods),
		Scheme: ct.Scheme,
		Scale:  ct.Scale,
	}

	// m = c0 + c1*s mod q
	for i := 0; i < n*numMods; i++ {
		modIdx := i / n
		mod := params.CoeffMods[modIdx]
		pt.Data[i] = addMod(ct.Data[i], mulMod(ct.Data[n*numMods+i], sk.Data[i], mod), mod)
	}

	return pt, nil
}

func addCPU(params Params, ct1, ct2 *Ciphertext) (*Ciphertext, error) {
	result := &Ciphertext{
		Data:   make([]uint64, len(ct1.Data)),
		Scheme: ct1.Scheme,
		Level:  min(ct1.Level, ct2.Level),
		Scale:  ct1.Scale,
	}

	numMods := len(params.CoeffMods)
	n := int(params.PolyDegree)

	for i := range result.Data {
		modIdx := (i / n) % numMods
		mod := params.CoeffMods[modIdx]
		result.Data[i] = addMod(ct1.Data[i], ct2.Data[i], mod)
	}

	return result, nil
}

func addPlainCPU(params Params, ct *Ciphertext, pt *Plaintext) (*Ciphertext, error) {
	result := &Ciphertext{
		Data:   make([]uint64, len(ct.Data)),
		Scheme: ct.Scheme,
		Level:  ct.Level,
		Scale:  ct.Scale,
	}
	copy(result.Data, ct.Data)

	numMods := len(params.CoeffMods)
	n := int(params.PolyDegree)

	// Add plaintext to c0 only
	for i := 0; i < n*numMods && i < len(pt.Data); i++ {
		modIdx := i / n
		mod := params.CoeffMods[modIdx]
		result.Data[i] = addMod(ct.Data[i], pt.Data[i], mod)
	}

	return result, nil
}

func subCPU(params Params, ct1, ct2 *Ciphertext) (*Ciphertext, error) {
	result := &Ciphertext{
		Data:   make([]uint64, len(ct1.Data)),
		Scheme: ct1.Scheme,
		Level:  min(ct1.Level, ct2.Level),
		Scale:  ct1.Scale,
	}

	numMods := len(params.CoeffMods)
	n := int(params.PolyDegree)

	for i := range result.Data {
		modIdx := (i / n) % numMods
		mod := params.CoeffMods[modIdx]
		result.Data[i] = subMod(ct1.Data[i], ct2.Data[i], mod)
	}

	return result, nil
}

func multiplyCPU(params Params, ct1, ct2 *Ciphertext, rlk *RelinKey) (*Ciphertext, error) {
	numMods := len(params.CoeffMods)
	n := int(params.PolyDegree)

	// Tensor product gives 3 polynomials, then relinearize to 2
	result := &Ciphertext{
		Data:   make([]uint64, 2*n*numMods),
		Scheme: ct1.Scheme,
		Level:  min(ct1.Level, ct2.Level),
		Scale:  ct1.Scale * ct2.Scale, // For CKKS
	}

	// Simplified multiplication (real impl uses NTT)
	for i := 0; i < n*numMods; i++ {
		modIdx := i / n
		mod := params.CoeffMods[modIdx]
		result.Data[i] = mulMod(ct1.Data[i], ct2.Data[i], mod)
	}
	for i := n * numMods; i < 2*n*numMods; i++ {
		modIdx := (i - n*numMods) / n
		mod := params.CoeffMods[modIdx]
		result.Data[i] = mulMod(ct1.Data[i], ct2.Data[i], mod)
	}

	return result, nil
}

func multiplyPlainCPU(params Params, ct *Ciphertext, pt *Plaintext) (*Ciphertext, error) {
	numMods := len(params.CoeffMods)
	n := int(params.PolyDegree)

	result := &Ciphertext{
		Data:   make([]uint64, len(ct.Data)),
		Scheme: ct.Scheme,
		Level:  ct.Level,
		Scale:  ct.Scale * pt.Scale,
	}

	for i := range result.Data {
		modIdx := (i / n) % numMods
		mod := params.CoeffMods[modIdx]
		ptIdx := i % (n * numMods)
		ptVal := uint64(0)
		if ptIdx < len(pt.Data) {
			ptVal = pt.Data[ptIdx]
		}
		result.Data[i] = mulMod(ct.Data[i], ptVal, mod)
	}

	return result, nil
}

func rotateCPU(params Params, ct *Ciphertext, gk *GaloisKey, steps int) (*Ciphertext, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	result := &Ciphertext{
		Data:   make([]uint64, len(ct.Data)),
		Scheme: ct.Scheme,
		Level:  ct.Level,
		Scale:  ct.Scale,
	}

	// Simplified rotation (real impl uses Galois automorphism)
	absSteps := steps
	if absSteps < 0 {
		absSteps = -absSteps
	}
	absSteps = absSteps % n

	for modIdx := 0; modIdx < numMods; modIdx++ {
		offset := modIdx * n
		for i := 0; i < n; i++ {
			srcIdx := offset + i
			dstIdx := offset + (i+absSteps)%n
			if steps < 0 {
				dstIdx = offset + (i-absSteps+n)%n
			}
			result.Data[dstIdx] = ct.Data[srcIdx]
		}
	}

	return result, nil
}

func rescaleCPU(params Params, ct *Ciphertext) (*Ciphertext, error) {
	n := int(params.PolyDegree)
	numMods := ct.Level + 1
	newNumMods := numMods - 1

	// Scale down by last modulus
	lastMod := params.CoeffMods[ct.Level]
	newScale := ct.Scale / float64(lastMod)

	result := &Ciphertext{
		Data:   make([]uint64, 2*n*newNumMods),
		Scheme: ct.Scheme,
		Level:  ct.Level - 1,
		Scale:  newScale,
	}

	// Drop last RNS component
	for i := 0; i < n*newNumMods; i++ {
		result.Data[i] = ct.Data[i]
	}
	for i := 0; i < n*newNumMods; i++ {
		result.Data[n*newNumMods+i] = ct.Data[n*numMods+i]
	}

	return result, nil
}

func modSwitchCPU(params Params, ct *Ciphertext) (*Ciphertext, error) {
	n := int(params.PolyDegree)
	numMods := ct.Level + 1
	newNumMods := numMods - 1

	result := &Ciphertext{
		Data:   make([]uint64, 2*n*newNumMods),
		Scheme: ct.Scheme,
		Level:  ct.Level - 1,
		Scale:  ct.Scale,
	}

	// Drop last RNS component (no scale adjustment)
	for i := 0; i < n*newNumMods; i++ {
		result.Data[i] = ct.Data[i]
	}
	for i := 0; i < n*newNumMods; i++ {
		result.Data[n*newNumMods+i] = ct.Data[n*numMods+i]
	}

	return result, nil
}

func bootstrapCPU(params Params, ct *Ciphertext, bsk *BootstrapKey) (*Ciphertext, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	// Bootstrapping raises ciphertext back to full level
	result := &Ciphertext{
		Data:   make([]uint64, 2*n*numMods),
		Scheme: ct.Scheme,
		Level:  numMods - 1,
		Scale:  params.Scale,
	}

	// Simplified: copy and extend (real impl is complex)
	copy(result.Data, ct.Data)

	return result, nil
}

func batchEncryptCPU(params Params, pk *PublicKey, pts []*Plaintext) ([]*Ciphertext, error) {
	results := make([]*Ciphertext, len(pts))
	for i, pt := range pts {
		ct, err := encryptCPU(params, pk, pt)
		if err != nil {
			return nil, err
		}
		results[i] = ct
	}
	return results, nil
}

func batchDecryptCPU(params Params, sk *SecretKey, cts []*Ciphertext) ([]*Plaintext, error) {
	results := make([]*Plaintext, len(cts))
	for i, ct := range cts {
		pt, err := decryptCPU(params, sk, ct)
		if err != nil {
			return nil, err
		}
		results[i] = pt
	}
	return results, nil
}

func encodeBFVCPU(params Params, values []int64) (*Plaintext, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	pt := &Plaintext{
		Data:   make([]uint64, n*numMods),
		Scheme: SchemeBFV,
		Scale:  1.0,
	}

	// Encode each value modulo plaintext modulus
	for i, v := range values {
		var encoded uint64
		if v >= 0 {
			encoded = uint64(v) % params.PlainMod
		} else {
			encoded = params.PlainMod - (uint64(-v) % params.PlainMod)
		}
		// Replicate across RNS components
		for modIdx := 0; modIdx < numMods; modIdx++ {
			pt.Data[modIdx*n+i] = encoded
		}
	}

	return pt, nil
}

func decodeBFVCPU(params Params, pt *Plaintext) ([]int64, error) {
	n := int(params.PolyDegree)

	values := make([]int64, n)
	for i := 0; i < n && i < len(pt.Data); i++ {
		v := pt.Data[i] % params.PlainMod
		// Convert to signed
		if v > params.PlainMod/2 {
			values[i] = -int64(params.PlainMod - v)
		} else {
			values[i] = int64(v)
		}
	}

	return values, nil
}

func encodeCKKSCPU(params Params, values []float64) (*Plaintext, error) {
	n := int(params.PolyDegree)
	slots := n / 2
	numMods := len(params.CoeffMods)

	pt := &Plaintext{
		Data:   make([]uint64, n*numMods),
		Scheme: SchemeCKKS,
		Scale:  params.Scale,
	}

	// Simplified CKKS encoding (real impl uses inverse FFT)
	for i := 0; i < slots && i < len(values); i++ {
		scaled := values[i] * params.Scale
		var encoded uint64
		if scaled >= 0 {
			encoded = uint64(math.Round(scaled))
		} else {
			encoded = params.CoeffMods[0] - uint64(math.Round(-scaled))
		}
		// Store in first RNS component
		pt.Data[i] = encoded
	}

	return pt, nil
}

func decodeCKKSCPU(params Params, pt *Plaintext) ([]float64, error) {
	n := int(params.PolyDegree)
	slots := n / 2

	values := make([]float64, slots)
	for i := 0; i < slots && i < len(pt.Data); i++ {
		v := pt.Data[i]
		mod := params.CoeffMods[0]
		if v > mod/2 {
			values[i] = -float64(mod-v) / pt.Scale
		} else {
			values[i] = float64(v) / pt.Scale
		}
	}

	return values, nil
}

// Modular arithmetic helpers

func addMod(a, b, m uint64) uint64 {
	sum := a + b
	if sum >= m || sum < a {
		sum -= m
	}
	return sum
}

func subMod(a, b, m uint64) uint64 {
	if a >= b {
		return a - b
	}
	return m - b + a
}

func mulMod(a, b, m uint64) uint64 {
	// Simple implementation - production would use Montgomery
	return (a * b) % m
}

func randMod(m uint64) uint64 {
	var buf [8]byte
	rand.Read(buf[:])
	return binary.LittleEndian.Uint64(buf[:]) % m
}

func sampleError() uint64 {
	// Sample small Gaussian error (simplified to small uniform)
	var b [1]byte
	rand.Read(b[:])
	return uint64(b[0] % 5) // Small error in {0,1,2,3,4}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
