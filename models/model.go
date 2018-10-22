package models

import (
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/learn/parameters"
	"github.com/RenatoGeh/gospn/spn"
)

type Model interface {
	// LearnStructure generates only the structure.
	LearnStructure(D spn.Dataset, L []int, Sc map[int]*learn.Variable)
	// LearnGenerative generates the structure and then applies generative gradient descent learning
	// on the model.
	LearnGenerative(D spn.Dataset, L []int, Sc map[int]*learn.Variable)
	// LearnDiscriminative generates the structure and then applies discriminative gradient descent
	// learning on the model.
	LearnDiscriminative(D spn.Dataset, L []int, Sc map[int]*learn.Variable)
	// Infer takes an instance spn.VarSet and returns the prediction and its probabilities.
	Infer(I spn.VarSet) (int, []float64)
	// TestAccuracy runs an accuracy test with a test dataset (D, L).
	TestAccuracy(D spn.Dataset, L []int)
	// Save saves this model to a file.
	Save(filename string) error
}

var (
	defParam *parameters.P
)

func init() {
	defParam = parameters.New(true, false, 0.01, parameters.HardGD, 1.0, 1.0, 1, 0.01, 30)
}
