package models

import (
	"github.com/RenatoGeh/godrive/data"
	dataset "github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/learn/gens"
	"github.com/RenatoGeh/gospn/spn"
)

const (
	gClusters = -1
	gPval     = 0.01
	gEps      = 4
	gMp       = 4
)

// Gens learns an SPN by training on D, L and Sc using predefined parameters. It then returns the
// resulting SPN.
func Gens(D spn.Dataset, L []int, Sc map[int]*learn.Variable) spn.SPN {
	T := dataset.MergeLabel(D, L, Sc[data.ClassVarid])
	S := gens.Learn(Sc, T, gClusters, gPval, gEps, gMp)
	return S
}
