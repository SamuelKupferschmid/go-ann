package main

import (
	"fmt"
	"math/rand"
)

//Network represents a artificial neural network with backpropagation
type Network struct {
	weights [][][]float64
}

//GetDimensions get an array with the size for each layer
func (n *Network) GetDimensions() []int {
	res := make([]int, len(n.weights)+1)
	res[0] = len(n.weights[0][0])

	for i := range n.weights {
		res[i+1] = len(n.weights[i])
	}

	return res
}

func (n *Network) Predict(input []float64) []float64 {
	return []float64{}
}

//NewNetwork creates a new Network by given dimensions
func NewNetwork(dims []int) *Network {
	n := &Network{}
	r := rand.New(rand.NewSource(672))

	n.weights = make([][][]float64, len(dims)-1)
	for i, val := range dims {
		if i > 0 {
			prev := dims[i-1]
			n.weights[i-1] = make([][]float64, val)

			for j := 0; j < val; j++ {
				n.weights[i-1][j] = make([]float64, prev)
				for k := 0; k < val; k++ {
					n.weights[i-1][j][k] = r.Float64()
				}
			}
		}
	}

	fmt.Println(n.weights)

	return n
}
