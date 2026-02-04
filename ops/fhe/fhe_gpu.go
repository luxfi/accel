//go:build accel

package fhe

import (
	"unsafe"

	"github.com/luxfi/accel"
)

// GPU-accelerated FHE implementations.
// Uses kernels from fhe/ kernel collection.
// These dispatch to the accel library's GPU backends (CUDA, Metal, Vulkan)
// for parallel polynomial operations in RNS form.

func keyGenGPU(sess *accel.Session, params Params) (*SecretKey, *PublicKey, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	// Generate ternary secret key using GPU random sampling
	// Kernel: Uses crypto/sha256 for CSPRNG seeding
	skSize := n * numMods
	pkSize := 2 * n * numMods

	// Create output tensors
	skTensor, err := accel.NewTensor[uint64](sess, []int{skSize})
	if err != nil {
		return keyGenCPU(params)
	}
	defer skTensor.Close()

	pkTensor, err := accel.NewTensor[uint64](sess, []int{pkSize})
	if err != nil {
		return keyGenCPU(params)
	}
	defer pkTensor.Close()

	// For key generation, we use lattice operations for polynomial sampling
	// The NTT forward transform is used for key computation
	// Fall back to CPU as keygen is typically done once
	return keyGenCPU(params)
}

func genRelinKeyGPU(sess *accel.Session, params Params, sk *SecretKey) (*RelinKey, error) {
	// Relinearization key generation requires:
	// 1. Computing s^2 in NTT domain
	// 2. Gadget decomposition
	// 3. Encrypting decomposed components

	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	// Create tensor for s^2 computation
	skTensor, err := accel.NewTensorWithData[uint64](sess, []int{n * numMods}, sk.Data)
	if err != nil {
		return genRelinKeyCPU(params, sk)
	}
	defer skTensor.Close()

	sk2Tensor, err := accel.NewTensor[uint64](sess, []int{n * numMods})
	if err != nil {
		return genRelinKeyCPU(params, sk)
	}
	defer sk2Tensor.Close()

	// Compute s^2 using field multiplication
	err = sess.ZK().FieldMul(
		skTensor.Untyped(),
		skTensor.Untyped(),
		sk2Tensor.Untyped(),
		params.CoeffMods[0],
	)
	if err != nil {
		return genRelinKeyCPU(params, sk)
	}

	if err := sess.Sync(); err != nil {
		return genRelinKeyCPU(params, sk)
	}

	// Full relinkey generation is complex, use CPU
	return genRelinKeyCPU(params, sk)
}

func genGaloisKeyGPU(sess *accel.Session, params Params, sk *SecretKey, steps []int) (*GaloisKey, error) {
	// Galois key generation requires automorphism evaluation
	// Fall back to CPU for initial implementation
	return genGaloisKeyCPU(params, sk, steps)
}

func genBootstrapKeyGPU(sess *accel.Session, params Params, sk *SecretKey) (*BootstrapKey, error) {
	// Bootstrap key is very large and complex
	// Fall back to CPU for initial implementation
	return genBootstrapKeyCPU(params, sk)
}

func encryptGPU(sess *accel.Session, params Params, pk *PublicKey, pt *Plaintext) (*Ciphertext, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	// Create tensors for plaintext and public key
	ptTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(pt.Data)}, pt.Data)
	if err != nil {
		return encryptCPU(params, pk, pt)
	}
	defer ptTensor.Close()

	pkTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(pk.Data)}, pk.Data)
	if err != nil {
		return encryptCPU(params, pk, pt)
	}
	defer pkTensor.Close()

	// Output ciphertext tensor
	ctSize := 2 * n * numMods
	ctTensor, err := accel.NewTensor[uint64](sess, []int{ctSize})
	if err != nil {
		return encryptCPU(params, pk, pt)
	}
	defer ctTensor.Close()

	// Dispatch BFV/CKKS encryption kernel via FHE ops
	fheOps := sess.FHE()
	err = fheOps.BFVEncrypt(ptTensor.Untyped(), pkTensor.Untyped(), ctTensor.Untyped())
	if err != nil {
		return encryptCPU(params, pk, pt)
	}

	if err := sess.Sync(); err != nil {
		return encryptCPU(params, pk, pt)
	}

	// Read result
	ctData, err := ctTensor.ToSlice()
	if err != nil {
		return encryptCPU(params, pk, pt)
	}

	return &Ciphertext{
		Data:   ctData,
		Scheme: params.Scheme,
		Level:  numMods - 1,
		Scale:  pt.Scale,
	}, nil
}

func decryptGPU(sess *accel.Session, params Params, sk *SecretKey, ct *Ciphertext) (*Plaintext, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	// Create tensors
	ctTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(ct.Data)}, ct.Data)
	if err != nil {
		return decryptCPU(params, sk, ct)
	}
	defer ctTensor.Close()

	skTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(sk.Data)}, sk.Data)
	if err != nil {
		return decryptCPU(params, sk, ct)
	}
	defer skTensor.Close()

	ptSize := n * numMods
	ptTensor, err := accel.NewTensor[uint64](sess, []int{ptSize})
	if err != nil {
		return decryptCPU(params, sk, ct)
	}
	defer ptTensor.Close()

	// Dispatch decryption kernel
	err = sess.FHE().BFVDecrypt(ctTensor.Untyped(), skTensor.Untyped(), ptTensor.Untyped())
	if err != nil {
		return decryptCPU(params, sk, ct)
	}

	if err := sess.Sync(); err != nil {
		return decryptCPU(params, sk, ct)
	}

	ptData, err := ptTensor.ToSlice()
	if err != nil {
		return decryptCPU(params, sk, ct)
	}

	return &Plaintext{
		Data:   ptData,
		Scheme: ct.Scheme,
		Scale:  ct.Scale,
	}, nil
}

func addGPU(sess *accel.Session, params Params, ct1, ct2 *Ciphertext) (*Ciphertext, error) {
	// Ciphertext addition is component-wise modular addition
	// This is highly parallelizable

	// Create tensors
	ct1Tensor, err := accel.NewTensorWithData[uint64](sess, []int{len(ct1.Data)}, ct1.Data)
	if err != nil {
		return addCPU(params, ct1, ct2)
	}
	defer ct1Tensor.Close()

	ct2Tensor, err := accel.NewTensorWithData[uint64](sess, []int{len(ct2.Data)}, ct2.Data)
	if err != nil {
		return addCPU(params, ct1, ct2)
	}
	defer ct2Tensor.Close()

	resultTensor, err := accel.NewTensor[uint64](sess, []int{len(ct1.Data)})
	if err != nil {
		return addCPU(params, ct1, ct2)
	}
	defer resultTensor.Close()

	// Dispatch FHE add kernel
	err = sess.FHE().BFVAdd(ct1Tensor.Untyped(), ct2Tensor.Untyped(), resultTensor.Untyped())
	if err != nil {
		return addCPU(params, ct1, ct2)
	}

	if err := sess.Sync(); err != nil {
		return addCPU(params, ct1, ct2)
	}

	resultData, err := resultTensor.ToSlice()
	if err != nil {
		return addCPU(params, ct1, ct2)
	}

	return &Ciphertext{
		Data:   resultData,
		Scheme: ct1.Scheme,
		Level:  minInt(ct1.Level, ct2.Level),
		Scale:  ct1.Scale,
	}, nil
}

func addPlainGPU(sess *accel.Session, params Params, ct *Ciphertext, pt *Plaintext) (*Ciphertext, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	// Create tensors
	ctTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(ct.Data)}, ct.Data)
	if err != nil {
		return addPlainCPU(params, ct, pt)
	}
	defer ctTensor.Close()

	// Pad plaintext to match ciphertext size if needed
	ptData := make([]uint64, n*numMods)
	copy(ptData, pt.Data)
	ptTensor, err := accel.NewTensorWithData[uint64](sess, []int{n * numMods}, ptData)
	if err != nil {
		return addPlainCPU(params, ct, pt)
	}
	defer ptTensor.Close()

	resultTensor, err := accel.NewTensor[uint64](sess, []int{len(ct.Data)})
	if err != nil {
		return addPlainCPU(params, ct, pt)
	}
	defer resultTensor.Close()

	// Use field add on first component (c0 + m)
	err = sess.ZK().FieldAdd(
		ctTensor.Untyped(),
		ptTensor.Untyped(),
		resultTensor.Untyped(),
		params.CoeffMods[0],
	)
	if err != nil {
		return addPlainCPU(params, ct, pt)
	}

	if err := sess.Sync(); err != nil {
		return addPlainCPU(params, ct, pt)
	}

	resultData, err := resultTensor.ToSlice()
	if err != nil {
		return addPlainCPU(params, ct, pt)
	}

	return &Ciphertext{
		Data:   resultData,
		Scheme: ct.Scheme,
		Level:  ct.Level,
		Scale:  ct.Scale,
	}, nil
}

func subGPU(sess *accel.Session, params Params, ct1, ct2 *Ciphertext) (*Ciphertext, error) {
	// Subtraction via negation and addition
	// c1 - c2 = c1 + (-c2)

	ct1Tensor, err := accel.NewTensorWithData[uint64](sess, []int{len(ct1.Data)}, ct1.Data)
	if err != nil {
		return subCPU(params, ct1, ct2)
	}
	defer ct1Tensor.Close()

	// Negate ct2: -x mod q = q - x
	negCt2Data := make([]uint64, len(ct2.Data))
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)
	for i := range ct2.Data {
		modIdx := (i / n) % numMods
		mod := params.CoeffMods[modIdx]
		if ct2.Data[i] == 0 {
			negCt2Data[i] = 0
		} else {
			negCt2Data[i] = mod - ct2.Data[i]
		}
	}

	negCt2Tensor, err := accel.NewTensorWithData[uint64](sess, []int{len(negCt2Data)}, negCt2Data)
	if err != nil {
		return subCPU(params, ct1, ct2)
	}
	defer negCt2Tensor.Close()

	resultTensor, err := accel.NewTensor[uint64](sess, []int{len(ct1.Data)})
	if err != nil {
		return subCPU(params, ct1, ct2)
	}
	defer resultTensor.Close()

	// Add ct1 + (-ct2)
	err = sess.FHE().BFVAdd(ct1Tensor.Untyped(), negCt2Tensor.Untyped(), resultTensor.Untyped())
	if err != nil {
		return subCPU(params, ct1, ct2)
	}

	if err := sess.Sync(); err != nil {
		return subCPU(params, ct1, ct2)
	}

	resultData, err := resultTensor.ToSlice()
	if err != nil {
		return subCPU(params, ct1, ct2)
	}

	return &Ciphertext{
		Data:   resultData,
		Scheme: ct1.Scheme,
		Level:  minInt(ct1.Level, ct2.Level),
		Scale:  ct1.Scale,
	}, nil
}

func multiplyGPU(sess *accel.Session, params Params, ct1, ct2 *Ciphertext, rlk *RelinKey) (*Ciphertext, error) {
	// FHE multiplication is the most computationally intensive operation
	// It consists of:
	// 1. Tensor product of polynomials (NTT-based)
	// 2. Relinearization to reduce ciphertext size

	ct1Tensor, err := accel.NewTensorWithData[uint64](sess, []int{len(ct1.Data)}, ct1.Data)
	if err != nil {
		return multiplyCPU(params, ct1, ct2, rlk)
	}
	defer ct1Tensor.Close()

	ct2Tensor, err := accel.NewTensorWithData[uint64](sess, []int{len(ct2.Data)}, ct2.Data)
	if err != nil {
		return multiplyCPU(params, ct1, ct2, rlk)
	}
	defer ct2Tensor.Close()

	rlkTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(rlk.Data)}, rlk.Data)
	if err != nil {
		return multiplyCPU(params, ct1, ct2, rlk)
	}
	defer rlkTensor.Close()

	resultTensor, err := accel.NewTensor[uint64](sess, []int{len(ct1.Data)})
	if err != nil {
		return multiplyCPU(params, ct1, ct2, rlk)
	}
	defer resultTensor.Close()

	// Dispatch FHE multiply kernel with relinearization
	err = sess.FHE().BFVMultiply(
		ct1Tensor.Untyped(),
		ct2Tensor.Untyped(),
		rlkTensor.Untyped(),
		resultTensor.Untyped(),
	)
	if err != nil {
		return multiplyCPU(params, ct1, ct2, rlk)
	}

	if err := sess.Sync(); err != nil {
		return multiplyCPU(params, ct1, ct2, rlk)
	}

	resultData, err := resultTensor.ToSlice()
	if err != nil {
		return multiplyCPU(params, ct1, ct2, rlk)
	}

	newScale := ct1.Scale
	if ct1.Scheme == SchemeCKKS {
		newScale = ct1.Scale * ct2.Scale
	}

	return &Ciphertext{
		Data:   resultData,
		Scheme: ct1.Scheme,
		Level:  minInt(ct1.Level, ct2.Level),
		Scale:  newScale,
	}, nil
}

func multiplyPlainGPU(sess *accel.Session, params Params, ct *Ciphertext, pt *Plaintext) (*Ciphertext, error) {
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	ctTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(ct.Data)}, ct.Data)
	if err != nil {
		return multiplyPlainCPU(params, ct, pt)
	}
	defer ctTensor.Close()

	// Pad plaintext
	ptData := make([]uint64, n*numMods)
	copy(ptData, pt.Data)
	ptTensor, err := accel.NewTensorWithData[uint64](sess, []int{n * numMods}, ptData)
	if err != nil {
		return multiplyPlainCPU(params, ct, pt)
	}
	defer ptTensor.Close()

	resultTensor, err := accel.NewTensor[uint64](sess, []int{len(ct.Data)})
	if err != nil {
		return multiplyPlainCPU(params, ct, pt)
	}
	defer resultTensor.Close()

	// Dispatch multiply plain kernel
	err = sess.FHE().BFVMultiplyPlain(ctTensor.Untyped(), ptTensor.Untyped(), resultTensor.Untyped())
	if err != nil {
		return multiplyPlainCPU(params, ct, pt)
	}

	if err := sess.Sync(); err != nil {
		return multiplyPlainCPU(params, ct, pt)
	}

	resultData, err := resultTensor.ToSlice()
	if err != nil {
		return multiplyPlainCPU(params, ct, pt)
	}

	return &Ciphertext{
		Data:   resultData,
		Scheme: ct.Scheme,
		Level:  ct.Level,
		Scale:  ct.Scale * pt.Scale,
	}, nil
}

func rotateGPU(sess *accel.Session, params Params, ct *Ciphertext, gk *GaloisKey, steps int) (*Ciphertext, error) {
	ctTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(ct.Data)}, ct.Data)
	if err != nil {
		return rotateCPU(params, ct, gk, steps)
	}
	defer ctTensor.Close()

	gkTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(gk.Data)}, gk.Data)
	if err != nil {
		return rotateCPU(params, ct, gk, steps)
	}
	defer gkTensor.Close()

	resultTensor, err := accel.NewTensor[uint64](sess, []int{len(ct.Data)})
	if err != nil {
		return rotateCPU(params, ct, gk, steps)
	}
	defer resultTensor.Close()

	// Dispatch rotation kernel
	err = sess.FHE().BFVRotate(ctTensor.Untyped(), gkTensor.Untyped(), steps, resultTensor.Untyped())
	if err != nil {
		return rotateCPU(params, ct, gk, steps)
	}

	if err := sess.Sync(); err != nil {
		return rotateCPU(params, ct, gk, steps)
	}

	resultData, err := resultTensor.ToSlice()
	if err != nil {
		return rotateCPU(params, ct, gk, steps)
	}

	return &Ciphertext{
		Data:   resultData,
		Scheme: ct.Scheme,
		Level:  ct.Level,
		Scale:  ct.Scale,
	}, nil
}

func rescaleGPU(sess *accel.Session, params Params, ct *Ciphertext) (*Ciphertext, error) {
	ctTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(ct.Data)}, ct.Data)
	if err != nil {
		return rescaleCPU(params, ct)
	}
	defer ctTensor.Close()

	// Result has one fewer RNS component
	n := int(params.PolyDegree)
	newNumMods := ct.Level
	resultSize := 2 * n * newNumMods

	resultTensor, err := accel.NewTensor[uint64](sess, []int{resultSize})
	if err != nil {
		return rescaleCPU(params, ct)
	}
	defer resultTensor.Close()

	// Dispatch CKKS rescale kernel
	err = sess.FHE().CKKSRescale(ctTensor.Untyped(), resultTensor.Untyped())
	if err != nil {
		return rescaleCPU(params, ct)
	}

	if err := sess.Sync(); err != nil {
		return rescaleCPU(params, ct)
	}

	resultData, err := resultTensor.ToSlice()
	if err != nil {
		return rescaleCPU(params, ct)
	}

	lastMod := params.CoeffMods[ct.Level]
	newScale := ct.Scale / float64(lastMod)

	return &Ciphertext{
		Data:   resultData,
		Scheme: ct.Scheme,
		Level:  ct.Level - 1,
		Scale:  newScale,
	}, nil
}

func modSwitchGPU(sess *accel.Session, params Params, ct *Ciphertext) (*Ciphertext, error) {
	// Modulus switching drops the last RNS component without rescaling
	// This is simpler than rescale - just truncate
	return modSwitchCPU(params, ct)
}

func bootstrapGPU(sess *accel.Session, params Params, ct *Ciphertext, bsk *BootstrapKey) (*Ciphertext, error) {
	ctTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(ct.Data)}, ct.Data)
	if err != nil {
		return bootstrapCPU(params, ct, bsk)
	}
	defer ctTensor.Close()

	bskTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(bsk.Data)}, bsk.Data)
	if err != nil {
		return bootstrapCPU(params, ct, bsk)
	}
	defer bskTensor.Close()

	// Result is at full level
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)
	resultSize := 2 * n * numMods

	resultTensor, err := accel.NewTensor[uint64](sess, []int{resultSize})
	if err != nil {
		return bootstrapCPU(params, ct, bsk)
	}
	defer resultTensor.Close()

	// Dispatch bootstrap kernel
	err = sess.FHE().Bootstrap(ctTensor.Untyped(), bskTensor.Untyped(), resultTensor.Untyped())
	if err != nil {
		return bootstrapCPU(params, ct, bsk)
	}

	if err := sess.Sync(); err != nil {
		return bootstrapCPU(params, ct, bsk)
	}

	resultData, err := resultTensor.ToSlice()
	if err != nil {
		return bootstrapCPU(params, ct, bsk)
	}

	return &Ciphertext{
		Data:   resultData,
		Scheme: ct.Scheme,
		Level:  numMods - 1,
		Scale:  params.Scale,
	}, nil
}

func batchEncryptGPU(sess *accel.Session, params Params, pk *PublicKey, pts []*Plaintext) ([]*Ciphertext, error) {
	m := len(pts)
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	// Pack all plaintexts into a single tensor
	ptSize := n * numMods
	allPtData := make([]uint64, m*ptSize)
	for i, pt := range pts {
		copy(allPtData[i*ptSize:(i+1)*ptSize], pt.Data)
	}

	ptTensor, err := accel.NewTensorWithData[uint64](sess, []int{m, ptSize}, allPtData)
	if err != nil {
		return batchEncryptCPU(params, pk, pts)
	}
	defer ptTensor.Close()

	pkTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(pk.Data)}, pk.Data)
	if err != nil {
		return batchEncryptCPU(params, pk, pts)
	}
	defer pkTensor.Close()

	ctSize := 2 * n * numMods
	ctTensor, err := accel.NewTensor[uint64](sess, []int{m, ctSize})
	if err != nil {
		return batchEncryptCPU(params, pk, pts)
	}
	defer ctTensor.Close()

	// Dispatch batch encryption kernel
	err = sess.FHE().BFVEncryptBatch(ptTensor.Untyped(), pkTensor.Untyped(), ctTensor.Untyped())
	if err != nil {
		return batchEncryptCPU(params, pk, pts)
	}

	if err := sess.Sync(); err != nil {
		return batchEncryptCPU(params, pk, pts)
	}

	allCtData, err := ctTensor.ToSlice()
	if err != nil {
		return batchEncryptCPU(params, pk, pts)
	}

	// Unpack results
	results := make([]*Ciphertext, m)
	for i := 0; i < m; i++ {
		ctData := make([]uint64, ctSize)
		copy(ctData, allCtData[i*ctSize:(i+1)*ctSize])
		results[i] = &Ciphertext{
			Data:   ctData,
			Scheme: params.Scheme,
			Level:  numMods - 1,
			Scale:  pts[i].Scale,
		}
	}

	return results, nil
}

func batchDecryptGPU(sess *accel.Session, params Params, sk *SecretKey, cts []*Ciphertext) ([]*Plaintext, error) {
	m := len(cts)
	n := int(params.PolyDegree)
	numMods := len(params.CoeffMods)

	// Pack all ciphertexts
	ctSize := 2 * n * numMods
	allCtData := make([]uint64, m*ctSize)
	for i, ct := range cts {
		copy(allCtData[i*ctSize:(i+1)*ctSize], ct.Data)
	}

	ctTensor, err := accel.NewTensorWithData[uint64](sess, []int{m, ctSize}, allCtData)
	if err != nil {
		return batchDecryptCPU(params, sk, cts)
	}
	defer ctTensor.Close()

	skTensor, err := accel.NewTensorWithData[uint64](sess, []int{len(sk.Data)}, sk.Data)
	if err != nil {
		return batchDecryptCPU(params, sk, cts)
	}
	defer skTensor.Close()

	ptSize := n * numMods
	ptTensor, err := accel.NewTensor[uint64](sess, []int{m, ptSize})
	if err != nil {
		return batchDecryptCPU(params, sk, cts)
	}
	defer ptTensor.Close()

	// Dispatch batch decryption - not directly supported, iterate
	// This could be optimized with a proper batch kernel
	results := make([]*Plaintext, m)
	for i, ct := range cts {
		pt, err := decryptGPU(sess, params, sk, ct)
		if err != nil {
			return batchDecryptCPU(params, sk, cts)
		}
		results[i] = pt
	}

	return results, nil
}

// Helper functions

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func uint64SliceToBytes(data []uint64) []byte {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*8)
}
