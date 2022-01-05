package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

/*
func main() {
	x := mat.NewDense(2, 2, []float64{1.0, 2.0, 3.0, 4.0})
	var y, z mat.VecDense

	fmt.Printf("Original:\n")
	fx := mat.Formatted(x, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("x = %v\n", fx)
	fmt.Printf("After scaling y:\n")
	y.ScaleVec(2.0, x.ColView(1))
	fx = mat.Formatted(x, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("x = %v\n", fx)
	fmt.Printf("After adding y to z:\n")
	z.AddVec(x.ColView(1), x.ColView(1))
	fx = mat.Formatted(x, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("x = %v\n", fx)
}
*/
