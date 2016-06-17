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
