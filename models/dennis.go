package models

import (
	"encoding/binary"
	"fmt"
	"github.com/RenatoGeh/gospn/conc"
	dataset "github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/learn/dennis"
	"github.com/RenatoGeh/gospn/learn/parameters"
	"github.com/RenatoGeh/gospn/score"
	"github.com/RenatoGeh/gospn/spn"
	"github.com/RenatoGeh/gospn/utils"
	"io/ioutil"
	"math"
	"os"
	"sync"
)

const (
	dClusters   = 1
	dSumsRegion = 4
	dGaussPixel = 4
	dSimThresh  = 0.975
)

// DVModel is an SPN based on the Dennis-Ventura algorithm (gospn/learn/dennis), but optimized for
// parallel programming. We assume that we have at least 4 CPU cores (e.g. Raspberry Pi B+).
type DVModel struct {
	S     []spn.SPN       // Each sub-SPN S[i] is restricted to label i.
	Y     *learn.Variable // Y is the query variable. Namely the direction variable.
	T     []*spn.Storer   // DP tables must be disjoint to be parallelized.
	procs int             // Number of processes to use for inference.
}

// NewDVModel creates a new DVModel.
func NewDVModel(Y *learn.Variable) *DVModel {
	S := make([]spn.SPN, Y.Categories)
	T := make([]*spn.Storer, Y.Categories)
	for i := 0; i < Y.Categories; i++ {
		T[i] = spn.NewStorer()
		T[i].NewTicket()
	}
	return &DVModel{S, Y, T, 3}
}

// LearnStructure learns only the DV structure.
func (M *DVModel) LearnStructure(D spn.Dataset, L []int, Sc map[int]*learn.Variable) {
	Q := conc.NewSingleQueue(-1)
	mu := &sync.Mutex{}

	cv := M.Y
	c := cv.Categories
	K := dataset.Split(D, c, L)
	S := M.S
	for i := 0; i < c; i++ {
		if len(K[i]) <= 0 {
			continue
		}
		Q.Run(func(id int) {
			mu.Lock()
			lsc := make(map[int]*learn.Variable)
			for k, v := range Sc {
				if k != cv.Varid {
					lsc[k] = v
				}
			}
			mu.Unlock()
			Z := dennis.Structure(K[id], lsc, dClusters, dSumsRegion, dGaussPixel, dSimThresh)
			S[id] = Z
		}, i)
	}
	Q.Wait()
}

// LearnGenerative fits the DVModel to the dataset (D, L) and scope Sc.
func (M *DVModel) LearnGenerative(D spn.Dataset, L []int, Sc map[int]*learn.Variable) {
	Q := conc.NewSingleQueue(-1)
	mu := &sync.Mutex{}

	cv := M.Y
	c := cv.Categories
	K := dataset.Split(D, c, L)
	S := M.S
	for i := 0; i < c; i++ {
		if len(K[i]) <= 0 {
			continue
		}
		Q.Run(func(id int) {
			mu.Lock()
			lsc := make(map[int]*learn.Variable)
			for k, v := range Sc {
				if k != cv.Varid {
					lsc[k] = v
				}
			}
			mu.Unlock()
			Z := dennis.Structure(K[id], lsc, dClusters, dSumsRegion, dGaussPixel, dSimThresh)
			parameters.Bind(Z, defParam)
			learn.Generative(Z, K[id])
			S[id] = Z
		}, i)
	}
	Q.Wait()
}

// LearnDiscriminative fits the DVModel to the dataset (D, L) and scope Sc.
func (M *DVModel) LearnDiscriminative(D spn.Dataset, L []int, Sc map[int]*learn.Variable) {
	Q := conc.NewSingleQueue(-1)
	mu := &sync.Mutex{}

	cv := M.Y
	c := cv.Categories
	K := dataset.Split(D, c, L)
	S := M.S
	for i := 0; i < c; i++ {
		if len(K[i]) <= 0 {
			continue
		}
		Q.Run(func(id int) {
			mu.Lock()
			lsc := make(map[int]*learn.Variable)
			for k, v := range Sc {
				if k != cv.Varid {
					lsc[k] = v
				}
			}
			mu.Unlock()
			Z := dennis.Structure(K[id], lsc, dClusters, dSumsRegion, dGaussPixel, dSimThresh)
			parameters.Bind(Z, defParam)
			learn.Discriminative(Z, K[id], []*learn.Variable{cv})
			S[id] = Z
		}, i)
	}
	Q.Wait()
}

// Infer takes an instance X and returns argmax_y P(Y=y|X), where Y is the query variable set on
// construction. Returns the most probable label and its probability.
func (M *DVModel) Infer(I spn.VarSet) (int, []float64) {
	Q := conc.NewSingleQueue(M.procs)
	c := M.Y.Categories
	v := M.Y.Varid
	P := make([]float64, c)
	for i := 0; i < c; i++ {
		Q.Run(func(id int) {
			nI := make(map[int]int)
			for k, u := range I {
				nI[k] = u
			}
			Z := M.S[id]
			B := M.T[id]
			spn.StoreInference(Z, nI, 0, B)
			pe, _ := B.Single(0, Z)
			B.Reset(0)
			P[id] = pe
		}, i)
	}
	Q.Wait()
	pe := utils.LogSumExp(P)
	for i := 0; i < c; i++ {
		Q.Run(func(id int) {
			nI := make(map[int]int)
			for k, u := range I {
				nI[k] = u
			}
			nI[v] = id
			Z := M.S[id]
			B := M.T[id]
			spn.StoreInference(Z, nI, 0, B)
			p, _ := B.Single(0, Z)
			B.Reset(0)
			P[id] = p
		}, i)
	}
	Q.Wait()
	ml, mp := -1, math.Inf(-1)
	for i, pi := range P {
		p := pi - pe
		P[i] = p
		if p > mp {
			ml, mp = i, p
		}
	}
	return ml, P
}

func (M *DVModel) TestAccuracy(D spn.Dataset, L []int) {
	score := score.NewScore()
	v := M.Y.Varid
	for i, I := range D {
		u := I[v]
		delete(I, v)
		l, _ := M.Infer(I)
		score.Register(l, L[i])
		I[v] = u
	}
	fmt.Println(score)
}

func (M *DVModel) Save(filename string) error {
	f, err := os.Create(filename)
	defer f.Close()
	if err != nil {
		return err
	}
	f.Write([]byte{byte(M.Y.Categories), byte(M.procs)})
	for _, S := range M.S {
		bytes := spn.Marshal(S)
		var n uint64 = uint64(len(bytes))
		nb := make([]byte, 8)
		binary.LittleEndian.PutUint64(nb, n)
		bytes = append(nb, bytes...)
		f.Write(bytes)
	}
	bytes, err := M.Y.GobEncode()
	if err != nil {
		return err
	}
	f.Write(bytes)
	return nil
}

func LoadDVModel(filename string) (*DVModel, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	var n int = int(bytes[0])
	var procs int = int(bytes[1])
	M := &DVModel{}
	M.S = make([]spn.SPN, n)
	M.T = make([]*spn.Storer, n)
	M.procs = procs
	for i := 0; i < n; i++ {
		M.T[i] = spn.NewStorer()
		M.T[i].NewTicket()
	}
	bytes = bytes[2:]
	for i := 0; i < n; i++ {
		mb := bytes[:8]
		bytes = bytes[8:]
		m := binary.LittleEndian.Uint64(mb)
		ms := bytes[:m]
		bytes = bytes[m:]
		M.S[i] = spn.Unmarshal(ms)
	}
	M.Y = &learn.Variable{}
	err = M.Y.GobDecode(bytes)
	if err != nil {
		return nil, err
	}
	return M, nil
}
