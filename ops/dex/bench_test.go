package dex

import (
	"testing"
)

func BenchmarkAccelMatchOrdersCPU(b *testing.B) {
	// Build a book with 100 bids and 100 asks
	bids := make([]Order, 100)
	asks := make([]Order, 100)
	for i := 0; i < 100; i++ {
		bids[i] = Order{
			ID: uint64(i + 1), UserID: 1,
			Price: uint64(5000000000 - uint64(i)*100000), Quantity: 100000000,
			Remaining: 100000000, Side: Bid, Type: Limit,
		}
		asks[i] = Order{
			ID: uint64(i + 101), UserID: 2,
			Price: uint64(5000100000 + uint64(i)*100000), Quantity: 100000000,
			Remaining: 100000000, Side: Ask, Type: Limit,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Reset remaining
		for j := range asks {
			asks[j].Remaining = asks[j].Quantity
		}
		incoming := []Order{{
			ID: uint64(10000 + i), UserID: 3,
			Price: asks[0].Price, Quantity: 100000000,
			Remaining: 100000000, Side: Bid, Type: Limit,
		}}
		matchOrdersCPU(bids, asks, incoming)
	}
}

func BenchmarkAccelBatchMatchCPU(b *testing.B) {
	// 10 books, 1 incoming each
	orderBooks := make([][]Order, 10)
	incomingBatches := make([][]Order, 10)
	for i := 0; i < 10; i++ {
		book := make([]Order, 20)
		for j := 0; j < 10; j++ {
			book[j] = Order{
				ID: uint64(i*100 + j + 1), UserID: 1,
				Price: uint64(5000000000 - uint64(j)*100000), Quantity: 100000000,
				Remaining: 100000000, Side: Bid, Type: Limit,
			}
			book[j+10] = Order{
				ID: uint64(i*100 + j + 11), UserID: 2,
				Price: uint64(5000100000 + uint64(j)*100000), Quantity: 100000000,
				Remaining: 100000000, Side: Ask, Type: Limit,
			}
		}
		orderBooks[i] = book
		incomingBatches[i] = []Order{{
			ID: uint64(10000 + i), UserID: 3,
			Price: 5000100000, Quantity: 100000000,
			Remaining: 100000000, Side: Bid, Type: Limit,
		}}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Reset remaining
		for k := range orderBooks {
			for j := range orderBooks[k] {
				orderBooks[k][j].Remaining = orderBooks[k][j].Quantity
			}
		}
		batchMatchCPU(orderBooks, incomingBatches)
	}
}
