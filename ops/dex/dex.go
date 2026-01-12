// Package dex provides GPU-accelerated DEX operations.
//
// Uses dex_swap.* kernels from backend plugins for order matching
// and atomic swap execution.
package dex

import (
	"errors"

	"github.com/luxfi/accel"
)

// Side represents order side (bid/ask).
type Side uint8

const (
	Bid Side = 0
	Ask Side = 1
)

// OrderType represents the order type.
type OrderType uint8

const (
	Limit  OrderType = 0
	Market OrderType = 1
	IOC    OrderType = 2 // Immediate-or-Cancel
	FOK    OrderType = 3 // Fill-or-Kill
)

// Order represents a trading order.
type Order struct {
	ID        uint64
	UserID    uint64
	Price     uint64 // Fixed-point Q32.32 or similar
	Quantity  uint64
	Remaining uint64
	Timestamp uint64
	Side      Side
	Type      OrderType
}

// Trade represents an executed trade.
type Trade struct {
	ID          uint64
	MakerID     uint64
	TakerID     uint64
	MakerUserID uint64
	TakerUserID uint64
	Price       uint64
	Quantity    uint64
	Timestamp   uint64
	TakerSide   Side
}

// Level represents an aggregated order book level.
type Level struct {
	Price      uint64
	Quantity   uint64
	OrderCount uint32
}

var (
	ErrInvalidOrder = errors.New("dex: invalid order")
	ErrNoMatch      = errors.New("dex: no matching orders")
	ErrBookEmpty    = errors.New("dex: order book empty")
)

// MatchOrders matches incoming orders against an order book.
// Uses GPU acceleration for parallel price-time priority matching.
// Returns executed trades and updated orders.
func MatchOrders(bids, asks []Order, incoming []Order) ([]Trade, []Order, error) {
	if len(bids) == 0 && len(asks) == 0 {
		return nil, nil, ErrBookEmpty
	}

	sess, err := accel.NewSession()
	if err != nil {
		return matchOrdersCPU(bids, asks, incoming)
	}
	defer sess.Close()

	return matchOrdersGPU(sess, bids, asks, incoming)
}

// BatchMatch matches multiple order batches in parallel.
// Useful for high-frequency trading scenarios.
func BatchMatch(orderBooks [][]Order, incomingBatches [][]Order) ([][]Trade, error) {
	if len(orderBooks) != len(incomingBatches) {
		return nil, ErrInvalidOrder
	}

	sess, err := accel.NewSession()
	if err != nil {
		return batchMatchCPU(orderBooks, incomingBatches)
	}
	defer sess.Close()

	return batchMatchGPU(sess, orderBooks, incomingBatches)
}

// AggregateBook aggregates orders into price levels.
// GPU-accelerated for large order books.
func AggregateBook(orders []Order, numLevels int) (bids, asks []Level, err error) {
	sess, err := accel.NewSession()
	if err != nil {
		return aggregateBookCPU(orders, numLevels)
	}
	defer sess.Close()

	return aggregateBookGPU(sess, orders, numLevels)
}

// Swap executes an atomic swap between two parties.
// Uses dex_swap kernel for GPU-accelerated execution.
func Swap(
	makerAsset, takerAsset []byte,
	makerAmount, takerAmount uint64,
	makerSig, takerSig []byte,
) ([]byte, error) {
	sess, err := accel.NewSession()
	if err != nil {
		return swapCPU(makerAsset, takerAsset, makerAmount, takerAmount, makerSig, takerSig)
	}
	defer sess.Close()

	return swapGPU(sess, makerAsset, takerAsset, makerAmount, takerAmount, makerSig, takerSig)
}
