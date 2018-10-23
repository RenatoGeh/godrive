package data

import (
	"github.com/RenatoGeh/gospn/io"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/spn"
	"github.com/RenatoGeh/gospn/sys"
)

const (
	train_256 = "data/train.npy"
	train_128 = "data/train_7.npy"
	train_064 = "data/train_6.npy"
	train_032 = "data/train_5.npy"
	train_016 = "data/train_4.npy"
	train_008 = "data/train_3.npy"
	test_256  = "data/test.npy"
	test_128  = "data/test_7.npy"
	test_064  = "data/test_6.npy"
	test_032  = "data/test_5.npy"
	test_016  = "data/test_4.npy"
	test_008  = "data/test_3.npy"

	train_256_eq = "data/train_eq.npy"
	train_128_eq = "data/train_7_eq.npy"
	train_064_eq = "data/train_6_eq.npy"
	train_032_eq = "data/train_5_eq.npy"
	train_016_eq = "data/train_4_eq.npy"
	train_008_eq = "data/train_3_eq.npy"
	test_256_eq  = "data/test_eq.npy"
	test_128_eq  = "data/test_7_eq.npy"
	test_064_eq  = "data/test_6_eq.npy"
	test_032_eq  = "data/test_5_eq.npy"
	test_016_eq  = "data/test_4_eq.npy"
	test_008_eq  = "data/test_3_eq.npy"
)

var (
	train_data = train_008_eq
	test_data  = test_008_eq

	ClassVar *learn.Variable
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

func pullTrain(file string, n int) (spn.Dataset, []int) {
	r, err := io.NewNpyReader(file)
	if err != nil {
		panic(err)
	}
	defer r.Close()
	D, L, err := r.ReadBalanced(n, 3)
	if err != nil {
		panic(err)
	}
	return D, L
}

func pullDataTrainOnly(trainFile string, n, m int) (spn.Dataset, []int, spn.Dataset, []int) {
	r, err := io.NewNpyReader(trainFile)
	if err != nil {
		panic(err)
	}
	defer r.Close()

	D, L, err := r.ReadBalanced(n+m, 3)
	if err != nil {
		panic(err)
	}
	return D[:n], L[:n], D[n:], L[n:]
}

// Prepare retrieves n and m instances of the training and test set respectively. It returns images
// in spn.Dataset format and labels in a separate slice.
func PrepareSample(n int, m int) (spn.Dataset, []int, spn.Dataset, []int, map[int]*learn.Variable) {
	S := Prepare()
	D, L, E, U := pullData(train_data, test_data, n, m)
	//D, L, E, U := pullDataTrainOnly("data/train_3_eq.npy", n, m)
	return D, L, E, U, S
}

// Prepare returns the full scope.
func Prepare() map[int]*learn.Variable {
	k := Width * Height
	sys.Width, sys.Height, sys.Max = Width, Height, Max
	S := make(map[int]*learn.Variable)
	for i := 0; i < k; i++ {
		S[i] = &learn.Variable{Varid: i, Categories: Max, Name: ""}
	}
	ClassVar = &learn.Variable{Varid: k, Categories: 3, Name: "cmd"}
	S[k] = ClassVar
	return S
}

// PrepareTrain returns a training dataset and the full scope.
func PrepareTrain(n int) (spn.Dataset, []int, map[int]*learn.Variable) {
	S := Prepare()
	D, L := pullTrain(train_data, n)
	return D, L, S
}
