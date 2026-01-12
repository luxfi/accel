package dex

import (
	"sort"
)

// CPU fallback implementations

func matchOrdersCPU(bids, asks []Order, incoming []Order) ([]Trade, []Order, error) {
	trades := make([]Trade, 0)
	updated := make([]Order, 0)

	// Sort bids descending by price, asks ascending
	sortedBids := make([]Order, len(bids))
	copy(sortedBids, bids)
	sort.Slice(sortedBids, func(i, j int) bool {
		return sortedBids[i].Price > sortedBids[j].Price
	})

	sortedAsks := make([]Order, len(asks))
	copy(sortedAsks, asks)
	sort.Slice(sortedAsks, func(i, j int) bool {
		return sortedAsks[i].Price < sortedAsks[j].Price
	})

	for _, order := range incoming {
		remaining := order.Remaining
		if remaining == 0 {
			remaining = order.Quantity
		}

		if order.Side == Bid {
			// Match against asks
			for i := range sortedAsks {
				if remaining == 0 {
					break
				}
				ask := &sortedAsks[i]
				if ask.Remaining == 0 || ask.Price > order.Price {
					continue
				}

				matchQty := min(remaining, ask.Remaining)
				trade := Trade{
					MakerID:     ask.ID,
					TakerID:     order.ID,
					MakerUserID: ask.UserID,
					TakerUserID: order.UserID,
					Price:       ask.Price,
					Quantity:    matchQty,
					TakerSide:   Bid,
				}
				trades = append(trades, trade)

				remaining -= matchQty
				ask.Remaining -= matchQty
			}
		} else {
			// Match against bids
			for i := range sortedBids {
				if remaining == 0 {
					break
				}
				bid := &sortedBids[i]
				if bid.Remaining == 0 || bid.Price < order.Price {
					continue
				}

				matchQty := min(remaining, bid.Remaining)
				trade := Trade{
					MakerID:     bid.ID,
					TakerID:     order.ID,
					MakerUserID: bid.UserID,
					TakerUserID: order.UserID,
					Price:       bid.Price,
					Quantity:    matchQty,
					TakerSide:   Ask,
				}
				trades = append(trades, trade)

				remaining -= matchQty
				bid.Remaining -= matchQty
			}
		}

		if remaining > 0 && order.Type == Limit {
			order.Remaining = remaining
			updated = append(updated, order)
		}
	}

	return trades, updated, nil
}

func batchMatchCPU(orderBooks [][]Order, incomingBatches [][]Order) ([][]Trade, error) {
	results := make([][]Trade, len(orderBooks))
	for i := range orderBooks {
		bids := make([]Order, 0)
		asks := make([]Order, 0)
		for _, o := range orderBooks[i] {
			if o.Side == Bid {
				bids = append(bids, o)
			} else {
				asks = append(asks, o)
			}
		}
		trades, _, err := matchOrdersCPU(bids, asks, incomingBatches[i])
		if err != nil {
			return nil, err
		}
		results[i] = trades
	}
	return results, nil
}

func aggregateBookCPU(orders []Order, numLevels int) (bids, asks []Level, err error) {
	bidMap := make(map[uint64]*Level)
	askMap := make(map[uint64]*Level)

	for _, o := range orders {
		if o.Side == Bid {
			if l, ok := bidMap[o.Price]; ok {
				l.Quantity += o.Remaining
				l.OrderCount++
			} else {
				bidMap[o.Price] = &Level{Price: o.Price, Quantity: o.Remaining, OrderCount: 1}
			}
		} else {
			if l, ok := askMap[o.Price]; ok {
				l.Quantity += o.Remaining
				l.OrderCount++
			} else {
				askMap[o.Price] = &Level{Price: o.Price, Quantity: o.Remaining, OrderCount: 1}
			}
		}
	}

	for _, l := range bidMap {
		bids = append(bids, *l)
	}
	for _, l := range askMap {
		asks = append(asks, *l)
	}

	// Sort and limit
	sort.Slice(bids, func(i, j int) bool { return bids[i].Price > bids[j].Price })
	sort.Slice(asks, func(i, j int) bool { return asks[i].Price < asks[j].Price })

	if len(bids) > numLevels {
		bids = bids[:numLevels]
	}
	if len(asks) > numLevels {
		asks = asks[:numLevels]
	}

	return bids, asks, nil
}

func swapCPU(makerAsset, takerAsset []byte, makerAmount, takerAmount uint64, makerSig, takerSig []byte) ([]byte, error) {
	// CPU atomic swap verification and execution
	// In practice, this validates signatures and creates swap proof
	return nil, nil
}
