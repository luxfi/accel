//go:build !accel

package dex

import (
	"github.com/luxfi/accel"
)

// When accel build tag is not set, GPU functions fall back to CPU

func matchOrdersGPU(sess *accel.Session, bids, asks []Order, incoming []Order) ([]Trade, []Order, error) {
	return matchOrdersCPU(bids, asks, incoming)
}

func batchMatchGPU(sess *accel.Session, orderBooks [][]Order, incomingBatches [][]Order) ([][]Trade, error) {
	return batchMatchCPU(orderBooks, incomingBatches)
}

func aggregateBookGPU(sess *accel.Session, orders []Order, numLevels int) (bids, asks []Level, err error) {
	return aggregateBookCPU(orders, numLevels)
}

func swapGPU(sess *accel.Session, makerAsset, takerAsset []byte, makerAmount, takerAmount uint64, makerSig, takerSig []byte) ([]byte, error) {
	return swapCPU(makerAsset, takerAsset, makerAmount, takerAmount, makerSig, takerSig)
}
