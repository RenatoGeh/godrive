package models

import (
	"fmt"
	"github.com/RenatoGeh/godrive/data"
	"github.com/RenatoGeh/gospn/conc"
	dataset "github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/learn/dennis"
	"github.com/RenatoGeh/gospn/learn/parameters"
	"github.com/RenatoGeh/gospn/score"
	"github.com/RenatoGeh/gospn/spn"
	"github.com/RenatoGeh/gospn/utils"
	"math"
	"sync"
)

const (
	dClusters   = 1
	dSumsRegion = 4
	dGaussPixel = 4
	dSimThresh  = 0.95
)

// DVModel is an SPN based on the Dennis-Ventura algorithm (gospn/learn/dennis), but optimized for
// parallel programming. We assume that we have at least 4 CPU cores (e.g. Raspberry Pi B+).
type DVModel struct {
	S []spn.SPN       // Each sub-SPN S[i] is restricted to label i.
	Y *learn.Variable // Y is the query variable. Namely the direction variable.
	T []*spn.Storer   // DP tables must be disjoint to be parallelized.
}

var (
	defParam *parameters.P
)

func init() {
	defParam = parameters.New(true, false, 0.01, parameters.HardGD, 0.01, 1.0, 0, 0.1, 4)
}

func GenerativeDennis(D spn.Dataset, L []int, Sc map[int]*learn.Variable) spn.SPN {
	Q := conc.NewSingleQueue(-1)
	mu := &sync.Mutex{}

	cv := Sc[data.ClassVarid]
	c := cv.Categories
	K := dataset.Split(D, c, L)
	root := spn.NewSum()
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
			S := dennis.Structure(K[id], lsc, dClusters, dSumsRegion, dGaussPixel, dSimThresh)
			parameters.Bind(S, defParam)
			learn.Generative(S, K[id])
			pi := spn.NewProduct()
			pi.AddChild(S)
			pi.AddChild(spn.NewIndicator(cv.Varid, id))
			mu.Lock()
			root.AddChildW(pi, 1.0/float64(c))
			mu.Unlock()
		}, i)
	}
	Q.Wait()
	return root
	//T := dataset.MergeLabel(D, L, Sc[data.ClassVarid])
	//S := dennis.Structure(T, Sc, dClusters, dSumsRegion, dGaussPixel, dSimThresh)
	//parameters.Bind(S, defParam)
	//learn.Generative(S, D)
	//return S
}

func DiscriminativeDennis(D spn.Dataset, L []int, Sc map[int]*learn.Variable) spn.SPN {
	Q := conc.NewSingleQueue(-1)
	mu := &sync.Mutex{}

	cv := Sc[data.ClassVarid]
	Y := []*learn.Variable{cv}
	c := cv.Categories
	K := dataset.Split(D, c, L)
	root := spn.NewSum()
	for i := 0; i < c; i++ {
		if len(K[i]) <= 0 {
			continue
		}
		Q.Run(func(id int) {
			mu.Lock()
			lsc := make(map[int]*learn.Variable)
			for k, v := range Sc {
				lsc[k] = v
			}
			mu.Unlock()
			S := dennis.Structure(K[id], lsc, dClusters, dSumsRegion, dGaussPixel, dSimThresh)
			parameters.Bind(S, defParam)
			learn.Discriminative(S, K[id], Y)
			pi := spn.NewProduct()
			pi.AddChild(S)
			pi.AddChild(spn.NewIndicator(cv.Varid, id))
			mu.Lock()
			root.AddChildW(pi, 1.0/float64(c))
			mu.Unlock()
		}, i)
	}
	Q.Wait()
	return root
}

// NewDVModel creates a new DVModel.
func NewDVModel(Y *learn.Variable) *DVModel {
	S := make([]spn.SPN, Y.Categories)
	T := make([]*spn.Storer, Y.Categories)
	for i := 0; i < Y.Categories; i++ {
		T[i] = spn.NewStorer()
		T[i].NewTicket()
	}
	return &DVModel{S, Y, T}
}

// Learn fits the DVModel to the dataset (D, L) and scope Sc.
func (M *DVModel) Learn(D spn.Dataset, L []int, Sc map[int]*learn.Variable) {
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
func (M *DVModel) Infer(I spn.VarSet) (int, float64) {
	Q := conc.NewSingleQueue(-1)
	mu := &sync.Mutex{}
	c := M.Y.Categories
	v := M.Y.Varid
	P := make([]float64, c)
	for i := 0; i < c; i++ {
		Q.Run(func(id int) {
			nI := make(map[int]int)
			mu.Lock()
			for k, v := range I {
				nI[k] = v
			}
			mu.Unlock()
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
			mu.Lock()
			for k, v := range I {
				nI[k] = v
			}
			mu.Unlock()
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
		if p := pi - pe; p > mp {
			ml, mp = i, p
		}
	}
	return ml, mp
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
