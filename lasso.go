package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"

	"math/rand"
)

func LASSO(X *mat.Dense, y *mat.VecDense, l float64, maxiter int) *mat.VecDense {
	var S0 float64
	ctr := 0
	_, cols := X.Dims()
	b := mat.NewVecDense(cols, nil)
	b1 := mat.NewVecDense(cols, nil)
	//initialise b with OLS estimator
	b.SolveVec(X, y)

	// Compute dot products used in updates
	xy := make([]float64, cols)
	for i := 0; i < cols; i++ {
		xy = append(xy, mat.Dot(X.ColView(i), y))
	}

	xx := mat.NewDense(cols, cols, nil)
	for i := 0; i < cols; i++ {
		for j := i; j < cols; j++ {
			xx.Set(i, j, mat.Dot(X.ColView(i), X.ColView(j)))
			xx.Set(j, i, xx.At(i, j))
		}
	}
	bc := mat.NewDense(cols, 1, nil)
	for ctr < maxiter {
		for p := 0; p < cols; p++ {
			bc.Set(p, 0, b.AtVec(p))
		}
		for j := 0; j < cols; j++ {
			bc.Set(j, 0, 0.0)
			S0 = 2.0 * (mat.Dot(xx.ColView(j), bc.ColView(0)) - xy[j])
			if S0 > l {
				b1.SetVec(j, (l-S0)/(2.0*xx.At(j, j)))
			}
			if S0 < -l {
				b1.SetVec(j, (-l-S0)/(2.0*xx.At(j, j)))
			}
			if (S0 >= -l) && (S0 <= l) {
				b1.SetVec(j, 0.0)
			}
		}
		ctr++
		fmt.Println(ctr)
		fmt.Printf("%v\n", b1)
		n := b.CopyVec(b1)
		if n < cols {
			fmt.Printf("Error copying\n")
		}
	}
	return b
}

func main() {
	//generate fake data
	r := rand.New(rand.NewSource(1234))
	d := make([]float64, 1000)
	for i := 0; i < 1000; i++ {
		d[i] = r.NormFloat64()
	}
	X := mat.NewDense(100, 10, d)
	y := mat.NewVecDense(100, nil)
	b := []float64{0.5, 0.0, 0.0, -1.0, 0.0, 0.3, 0.2, 0.8, 0.0, 0.0}
	for i := 0; i < 10; i++ {
		y.AddScaledVec(y, b[i], X.ColView(i))
	}
	noise := make([]float64, 100)
	for i := 0; i < 100; i++ {
		noise[i] = r.NormFloat64() * 0.001
	}
	y.AddVec(y, mat.NewVecDense(100, noise))

	var coeff mat.VecDense

	coeff = *LASSO(X, y, 0.1, 100)

	for i := 0; i < coeff.Len(); i++ {
		fmt.Printf("b[%d]:\t%v\n", i, coeff.AtVec(i))
	}
}
