package main

import (
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
