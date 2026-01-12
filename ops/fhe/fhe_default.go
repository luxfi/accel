//go:build !accel

package fhe

import (
	"github.com/luxfi/accel"
)

// When accel build tag is not set, GPU functions fall back to CPU implementations.

func keyGenGPU(sess *accel.Session, params Params) (*SecretKey, *PublicKey, error) {
	return keyGenCPU(params)
}

func genRelinKeyGPU(sess *accel.Session, params Params, sk *SecretKey) (*RelinKey, error) {
	return genRelinKeyCPU(params, sk)
}

func genGaloisKeyGPU(sess *accel.Session, params Params, sk *SecretKey, steps []int) (*GaloisKey, error) {
	return genGaloisKeyCPU(params, sk, steps)
}

func genBootstrapKeyGPU(sess *accel.Session, params Params, sk *SecretKey) (*BootstrapKey, error) {
	return genBootstrapKeyCPU(params, sk)
}

func encryptGPU(sess *accel.Session, params Params, pk *PublicKey, pt *Plaintext) (*Ciphertext, error) {
	return encryptCPU(params, pk, pt)
}

func decryptGPU(sess *accel.Session, params Params, sk *SecretKey, ct *Ciphertext) (*Plaintext, error) {
	return decryptCPU(params, sk, ct)
}

func addGPU(sess *accel.Session, params Params, ct1, ct2 *Ciphertext) (*Ciphertext, error) {
	return addCPU(params, ct1, ct2)
}

func addPlainGPU(sess *accel.Session, params Params, ct *Ciphertext, pt *Plaintext) (*Ciphertext, error) {
	return addPlainCPU(params, ct, pt)
}

func subGPU(sess *accel.Session, params Params, ct1, ct2 *Ciphertext) (*Ciphertext, error) {
	return subCPU(params, ct1, ct2)
}

func multiplyGPU(sess *accel.Session, params Params, ct1, ct2 *Ciphertext, rlk *RelinKey) (*Ciphertext, error) {
	return multiplyCPU(params, ct1, ct2, rlk)
}

func multiplyPlainGPU(sess *accel.Session, params Params, ct *Ciphertext, pt *Plaintext) (*Ciphertext, error) {
	return multiplyPlainCPU(params, ct, pt)
}

func rotateGPU(sess *accel.Session, params Params, ct *Ciphertext, gk *GaloisKey, steps int) (*Ciphertext, error) {
	return rotateCPU(params, ct, gk, steps)
}

func rescaleGPU(sess *accel.Session, params Params, ct *Ciphertext) (*Ciphertext, error) {
	return rescaleCPU(params, ct)
}

func modSwitchGPU(sess *accel.Session, params Params, ct *Ciphertext) (*Ciphertext, error) {
	return modSwitchCPU(params, ct)
}

func bootstrapGPU(sess *accel.Session, params Params, ct *Ciphertext, bsk *BootstrapKey) (*Ciphertext, error) {
	return bootstrapCPU(params, ct, bsk)
}

func batchEncryptGPU(sess *accel.Session, params Params, pk *PublicKey, pts []*Plaintext) ([]*Ciphertext, error) {
	return batchEncryptCPU(params, pk, pts)
}

func batchDecryptGPU(sess *accel.Session, params Params, sk *SecretKey, cts []*Ciphertext) ([]*Plaintext, error) {
	return batchDecryptCPU(params, sk, cts)
}
