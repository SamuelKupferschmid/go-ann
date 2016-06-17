package main

import (
	"math"
	"math/rand"
)

//Network represents a artificial neural network with backpropagation
type Network struct {
	weights [][][]float64
}

//GetDimensions get an array with the size for each layer
func (n *Network) GetDimensions() []int {
	res := make([]int, len(n.weights)+1)
	res[0] = len(n.weights[0][0]) - 1

	for i := range n.weights {
		res[i+1] = len(n.weights[i])
	}

	return res
}

//Predict predicts the outputs using the given input values and the weights
func (n *Network) Predict(input []float64) []float64 {
	var res []float64
	for _, layer := range n.weights {
		res = make([]float64, len(layer))
		for i, node := range layer {
			sum := node[0]

			for i := 1; i < len(node); i++ {
				sum += node[i] * input[i-1]
			}
			res[i] = sigmoid(sum)
		}
		input = res
	}
	return res
}

func sigmoid(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}

//GetWeights return a three dimemsional array with all weights
func (n *Network) GetWeights() [][][]float64 {
	return n.weights
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
				n.weights[i-1][j] = make([]float64, prev+1) //+1 for bias
				for k := 0; k <= prev; k++ {
					n.weights[i-1][j][k] = r.Float64()
				}
			}
		}
	}

	return n
}
