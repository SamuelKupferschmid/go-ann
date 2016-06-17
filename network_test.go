package main

import (
	"math"
	"reflect"
	"testing"
)

func TestSetupNetwork(t *testing.T) {

	dim := []int{2, 2, 1}
	n := NewNetwork(dim)

	if !reflect.DeepEqual(dim, n.GetDimensions()) {
		t.Error("dimension error")
	}
}

func TestPredictDimension(t *testing.T) {
	dim := []int{2, 1}
	n := NewNetwork(dim)

	in := []float64{0.4, 1}

	res := n.Predict(in)

	if len(res) != dim[1] {
		t.Error(("dimension error"))
	}
}

func TestDimension1221Bias(t *testing.T) {
	n := NewNetwork([]int{1, 2, 2, 1})
	in := []float64{0.2}
	n.Predict(in)
}

func TestPredictIsSigmoid(t *testing.T) {
	dim := []int{2, 1}
	n := NewNetwork(dim)

	in := []float64{99, 99}

	res := n.Predict(in)

	if res[0] <= 0 && res[0] >= 1 {
		t.Error("result not between 0 and 1")
	}
}

func TestSigmoidTest(t *testing.T) {
	if sigmoid(math.MaxFloat64) != 1 {
		t.Fail()
	}

	if sigmoid(0) != 0.5 {
		t.Fail()
	}

	if sigmoid(-math.MaxFloat64) <= math.SmallestNonzeroFloat64 {
		t.Fail()
	}

}

func TestNodesHasBias(t *testing.T) {
	dim := []int{2, 2, 1}
	n := NewNetwork(dim)
	w := n.GetWeights()

	if len(w[0][0]) != 3 {
		t.Errorf("3 weights expected %d given", len(w[0][0]))
	}
}

func TestOutputLayerHasNoBias(t *testing.T) {
	n := NewNetwork([]int{1, 3, 2, 1})

	if len(n.GetWeights()[2]) != 1 {
		t.Error("dimension error")
	}
}

func TestBackpropagateConverges(t *testing.T) {
	n := NewNetwork([]int{1, 2, 2, 1})
	in := []float64{0.4}
	target := []float64{0.7}

	delta := math.MaxFloat64

	for i := 0; i < 100; i++ {
		res := n.Predict(in)[0]

		if math.Abs(res-target[0]) >= delta {
			t.Error("error not shrinken while backpropagation")
			return
		}

		delta = math.Abs(res - target[0])
		n.Backpropagate([]float64{target[0] - res})
	}
}
