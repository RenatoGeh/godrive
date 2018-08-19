package main

import (
	"fmt"
	"github.com/RenatoGeh/godrive/data"
	"github.com/RenatoGeh/godrive/models"
	dataset "github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/io"
	"github.com/RenatoGeh/gospn/sys"
)

func main() {
	n, m := 300, 300
	R, L, T, U, S := data.Prepare(n, m)
	Z := models.Gens(R, L, S)
	models.Accuracy(Z, T, U, S)
	return
	DV := models.NewDVModel(S[data.ClassVarid])
	fmt.Println("Learning...")
	DV.Learn(R, L, S)
	fmt.Println("Testing...")
	DV.TestAccuracy(T, U)
	return
	sys.Verbose = false
	N := models.GenerativeDennis(R, L, S)
	models.Accuracy(N, T, U, S)
	return
	dirs := [3]string{"up", "left", "right"}
	K := dataset.Split(R, S[data.ClassVarid].Categories, L)
	for c, k := range K {
		for i, I := range k {
			io.VarSetToPGM(fmt.Sprintf("samples/%s/%d.pgm", dirs[c], i), I, data.Width, data.Height, data.Max-1)
		}
	}
}
