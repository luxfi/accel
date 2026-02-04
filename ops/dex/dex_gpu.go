//go:build accel

package dex

import (
	"encoding/binary"
	"unsafe"

	"github.com/luxfi/accel"
)

// GPU-accelerated DEX implementations.
// These dispatch to the accel library's GPU backends (CUDA, Metal, Vulkan)
// for parallel order matching, book aggregation, and atomic swaps.

// orderSize is the packed size of an Order in bytes for GPU transfer
// ID(8) + UserID(8) + Price(8) + Quantity(8) + Remaining(8) + Timestamp(8) + Side(1) + Type(1) + padding(6) = 56
const orderSize = 56

// tradeSize is the packed size of a Trade in bytes
// ID(8) + MakerID(8) + TakerID(8) + MakerUserID(8) + TakerUserID(8) + Price(8) + Quantity(8) + Timestamp(8) + TakerSide(1) + padding(7) = 72
const tradeSize = 72

// packOrders packs orders into a flat byte slice for GPU transfer
func packOrders(orders []Order) []byte {
	data := make([]byte, len(orders)*orderSize)
	for i, o := range orders {
		offset := i * orderSize
		binary.LittleEndian.PutUint64(data[offset:], o.ID)
		binary.LittleEndian.PutUint64(data[offset+8:], o.UserID)
		binary.LittleEndian.PutUint64(data[offset+16:], o.Price)
		binary.LittleEndian.PutUint64(data[offset+24:], o.Quantity)
		binary.LittleEndian.PutUint64(data[offset+32:], o.Remaining)
		binary.LittleEndian.PutUint64(data[offset+40:], o.Timestamp)
		data[offset+48] = byte(o.Side)
		data[offset+49] = byte(o.Type)
	}
	return data
}

// unpackTrades unpacks trades from a flat byte slice
func unpackTrades(data []byte, count int) []Trade {
	trades := make([]Trade, count)
	for i := 0; i < count; i++ {
		offset := i * tradeSize
		trades[i] = Trade{
			ID:          binary.LittleEndian.Uint64(data[offset:]),
			MakerID:     binary.LittleEndian.Uint64(data[offset+8:]),
			TakerID:     binary.LittleEndian.Uint64(data[offset+16:]),
			MakerUserID: binary.LittleEndian.Uint64(data[offset+24:]),
			TakerUserID: binary.LittleEndian.Uint64(data[offset+32:]),
			Price:       binary.LittleEndian.Uint64(data[offset+40:]),
			Quantity:    binary.LittleEndian.Uint64(data[offset+48:]),
			Timestamp:   binary.LittleEndian.Uint64(data[offset+56:]),
			TakerSide:   Side(data[offset+64]),
		}
	}
	return trades
}

func matchOrdersGPU(sess *accel.Session, bids, asks []Order, incoming []Order) ([]Trade, []Order, error) {
	// Order matching using GPU for parallel price comparison
	// The algorithm:
	// 1. Sort orders by price (bids desc, asks asc) - can be done on GPU
	// 2. For each incoming order, find best match using parallel search
	// 3. Execute matches and update remaining quantities

	if len(incoming) == 0 {
		return nil, nil, nil
	}

	// Pack orders for GPU
	bidData := packOrders(bids)
	askData := packOrders(asks)
	incomingData := packOrders(incoming)

	// Create tensors
	bidTensor, err := accel.NewTensorWithData[uint8](sess, []int{len(bids), orderSize}, bidData)
	if err != nil {
		return matchOrdersCPU(bids, asks, incoming)
	}
	defer bidTensor.Close()

	askTensor, err := accel.NewTensorWithData[uint8](sess, []int{len(asks), orderSize}, askData)
	if err != nil {
		return matchOrdersCPU(bids, asks, incoming)
	}
	defer askTensor.Close()

	incomingTensor, err := accel.NewTensorWithData[uint8](sess, []int{len(incoming), orderSize}, incomingData)
	if err != nil {
		return matchOrdersCPU(bids, asks, incoming)
	}
	defer incomingTensor.Close()

	// Output tensor for trades (max possible is len(incoming) * max(len(bids), len(asks)))
	maxTrades := len(incoming) * (len(bids) + len(asks))
	if maxTrades == 0 {
		maxTrades = 1
	}
	tradeTensor, err := accel.NewTensor[uint8](sess, []int{maxTrades, tradeSize})
	if err != nil {
		return matchOrdersCPU(bids, asks, incoming)
	}
	defer tradeTensor.Close()

	// Trade count output
	countTensor, err := accel.NewTensor[uint32](sess, []int{1})
	if err != nil {
		return matchOrdersCPU(bids, asks, incoming)
	}
	defer countTensor.Close()

	// Dispatch matching kernel via DEX ops
	err = sess.DEX().MatchOrders(
		bidTensor.Untyped(),
		askTensor.Untyped(),
		incomingTensor.Untyped(),
		tradeTensor.Untyped(),
		countTensor.Untyped(),
	)
	if err != nil {
		return matchOrdersCPU(bids, asks, incoming)
	}

	if err := sess.Sync(); err != nil {
		return matchOrdersCPU(bids, asks, incoming)
	}

	// Read trade count
	countData, err := countTensor.ToSlice()
	if err != nil {
		return matchOrdersCPU(bids, asks, incoming)
	}
	tradeCount := int(countData[0])

	if tradeCount == 0 {
		// No matches, return remaining orders
		updated := make([]Order, 0)
		for _, o := range incoming {
			if o.Type == Limit {
				if o.Remaining == 0 {
					o.Remaining = o.Quantity
				}
				updated = append(updated, o)
			}
		}
		return nil, updated, nil
	}

	// Read trades
	tradeData, err := tradeTensor.ToSlice()
	if err != nil {
		return matchOrdersCPU(bids, asks, incoming)
	}

	trades := unpackTrades(tradeData, tradeCount)

	// Compute remaining orders (unfilled limit orders)
	updated := make([]Order, 0)
	for _, o := range incoming {
		if o.Type == Limit {
			// Check if partially filled
			filled := uint64(0)
			for _, t := range trades {
				if t.TakerID == o.ID {
					filled += t.Quantity
				}
			}
			if filled < o.Quantity {
				o.Remaining = o.Quantity - filled
				updated = append(updated, o)
			}
		}
	}

	return trades, updated, nil
}

func batchMatchGPU(sess *accel.Session, orderBooks [][]Order, incomingBatches [][]Order) ([][]Trade, error) {
	// Batch matching processes multiple order books in parallel
	// Each order book is matched independently

	m := len(orderBooks)
	if m == 0 {
		return nil, nil
	}

	results := make([][]Trade, m)

	// Process each book - could be parallelized with a proper batch kernel
	for i := 0; i < m; i++ {
		bids := make([]Order, 0)
		asks := make([]Order, 0)
		for _, o := range orderBooks[i] {
			if o.Side == Bid {
				bids = append(bids, o)
			} else {
				asks = append(asks, o)
			}
		}

		trades, _, err := matchOrdersGPU(sess, bids, asks, incomingBatches[i])
		if err != nil {
			return batchMatchCPU(orderBooks, incomingBatches)
		}
		results[i] = trades
	}

	return results, nil
}

func aggregateBookGPU(sess *accel.Session, orders []Order, numLevels int) (bids, asks []Level, err error) {
	// Book aggregation - currently no dedicated GPU kernel in DEXOps interface
	// Fall back to CPU implementation which uses map-based aggregation
	// Future: implement using parallel reduction on GPU when kernel is available
	_ = sess
	return aggregateBookCPU(orders, numLevels)
}

func swapGPU(sess *accel.Session, makerAsset, takerAsset []byte, makerAmount, takerAmount uint64, makerSig, takerSig []byte) ([]byte, error) {
	// Atomic swap - no dedicated GPU kernel in DEXOps interface
	// The swap execution logic (signature verification) can use Crypto ops
	// but the swap itself requires CPU orchestration
	// Future: implement full swap kernel when available in DEXOps
	_ = sess
	return swapCPU(makerAsset, takerAsset, makerAmount, takerAmount, makerSig, takerSig)
}

// Helper to convert slice to byte slice
func uint64SliceToBytes(data []uint64) []byte {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*8)
}
