package data

import (
	"github.com/RenatoGeh/gospn/io"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/spn"
	"github.com/RenatoGeh/gospn/sys"
)

const (
	Width  = 80
	Height = 45
	Max    = 8

	ClassVarid = Width * Height
)

func pullData(trainFile, testFile string, n, m int) (spn.Dataset, []int, spn.Dataset, []int) {
	r, err := io.NewNpyReader(trainFile)
	if err != nil {
		panic(err)
	}
	t, err := io.NewNpyReader(testFile)
	if err != nil {
		panic(err)
	}
	defer r.Close()
	defer t.Close()

	D, L, err := r.ReadBalanced(n, 3)
	if err != nil {
		panic(err)
	}
	E, U, err := t.ReadBalanced(m, 3)
	if err != nil {
		panic(err)
	}
	return D, L, E, U
}

// Prepare retrieves n and m instances of the training and test set respectively. It returns images
// in spn.Dataset format and labels in a separate slice.
func Prepare(n int, m int) (spn.Dataset, []int, spn.Dataset, []int, map[int]*learn.Variable) {
	k := Width * Height
	S := make(map[int]*learn.Variable)
	for i := 0; i < k; i++ {
		S[i] = &learn.Variable{i, Max + 1, ""}
	}
	S[k] = &learn.Variable{k, 3, "cmd"}
	sys.Width, sys.Height, sys.Max = Width, Height, Max
	D, L, E, U := pullData("data/train_3.npy", "data/test_3.npy", n, m)
	return D, L, E, U, S
}
