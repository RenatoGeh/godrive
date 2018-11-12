package models

import (
	"encoding/binary"
	"fmt"
	"github.com/RenatoGeh/godrive/data"
	"github.com/RenatoGeh/gospn/conc"
	dataset "github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/learn/gens"
	"github.com/RenatoGeh/gospn/learn/parameters"
	"github.com/RenatoGeh/gospn/score"
	"github.com/RenatoGeh/gospn/spn"
	"github.com/RenatoGeh/gospn/utils"
	"io/ioutil"
	"math"
	"os"
)

const (
	gClusters = 2
	gPval     = 0.01
	gEps      = 4
	gMp       = 4
)

type GensModel struct {
	S     spn.SPN
	Y     *learn.Variable
	procs int
}

func NewGensModel(Y *learn.Variable) *GensModel {
	return &GensModel{nil, Y, 3}
}

func (M *GensModel) LearnStructure(D spn.Dataset, L []int, Sc map[int]*learn.Variable) {
	T := dataset.MergeLabel(D, L, data.ClassVar)
	M.S = gens.LearnConcurrent(Sc, T, gClusters, gPval, gEps, gMp, -1)
}

func (M *GensModel) LearnGenerative(D spn.Dataset, L []int, Sc map[int]*learn.Variable) {
	M.LearnStructure(D, L, Sc)
	parameters.Bind(M.S, defParam)
	T := dataset.MergeLabel(D, L, data.ClassVar)
	learn.Generative(M.S, T)
}

func (M *GensModel) LearnDiscriminative(D spn.Dataset, L []int, Sc map[int]*learn.Variable) {
	M.LearnStructure(D, L, Sc)
	parameters.Bind(M.S, defParam)
	T := dataset.MergeLabel(D, L, data.ClassVar)
	learn.Discriminative(M.S, T, []*learn.Variable{M.Y})
}

func (M *GensModel) Infer(I spn.VarSet) (int, []float64) {
	Q := conc.NewSingleQueue(M.procs)
	c := M.Y.Categories
	v := M.Y.Varid
	P := make([]float64, c)
	for i := 0; i < c; i++ {
		Q.Run(func(id int) {
			p := spn.InferenceY(M.S, I, v, id)
			P[id] = p
		}, i)
	}
	Q.Wait()
	pe := utils.LogSumExp(P)
	ml, mp := -1, math.Inf(-1)
	for i, p := range P {
		if p > mp {
			ml, mp = i, p
		}
		P[i] = math.Exp(p - pe)
	}
	return ml, P
}

func (M *GensModel) TestAccuracy(D spn.Dataset, L []int) {
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
	printCM(score.ConfusionMatrix(data.ClassVar.Categories), len(D))
}

func (M *GensModel) Save(filename string) error {
	f, err := os.Create(filename)
	defer f.Close()
	if err != nil {
		return err
	}
	f.Write([]byte{byte(M.Y.Categories), byte(M.procs)})
	bytes := spn.Marshal(M.S)
	var n uint64 = uint64(len(bytes))
	nb := make([]byte, 8)
	binary.LittleEndian.PutUint64(nb, n)
	bytes = append(nb, bytes...)
	f.Write(bytes)
	bytes, err = M.Y.GobEncode()
	if err != nil {
		return err
	}
	f.Write(bytes)
	return nil
}

func LoadGensModel(filename string) (*GensModel, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	procs := int(bytes[1])
	M := &GensModel{}
	M.procs = procs
	bytes = bytes[2:]
	m := binary.LittleEndian.Uint64(bytes[:8])
	bytes = bytes[8:]
	M.S = spn.Unmarshal(bytes[:m])
	bytes = bytes[m:]
	M.Y = &learn.Variable{}
	err = M.Y.GobDecode(bytes)
	if err != nil {
		return nil, err
	}
	return M, nil
}
