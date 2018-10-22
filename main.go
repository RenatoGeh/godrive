package main

import (
	"github.com/RenatoGeh/godrive/bot"
	"github.com/RenatoGeh/godrive/camera"
	"github.com/RenatoGeh/godrive/data"
	"github.com/RenatoGeh/godrive/models"
)

func Train(t, l string, n int, filename string) {
	var M models.Model
	if t == "dv" {
		M = models.NewDVModel(data.ClassVar)
	} else {
		M = models.NewGensModel(data.ClassVar)
	}

	D, L, Sc := data.PrepareTrain(n)
	if l == "gen" {
		M.LearnGenerative(D, L, Sc)
	} else if l == "disc" {
		M.LearnDiscriminative(D, L, Sc)
	} else {
		M.LearnStructure(D, L, Sc)
	}

	M.Save(filename)
}

func Test(t string) {
	var M models.Model
	var err error
	if t == "dv" {
		M, err = models.LoadDVModel("saved/dv.mdl")
	} else {
		M, err = models.LoadGensModel("saved/dv.mdl")
	}
	if err != nil {
		panic(err)
	}
	B, err := bot.New(0, M)
	if err != nil {
		panic(err)
	}
	defer B.Close()

	B.SetTransform(camera.MakeQuantize(8))
	B.Start()
}

func main() {
}
