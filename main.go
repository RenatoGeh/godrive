package main

import (
	"fmt"
	"github.com/RenatoGeh/godrive/bot"
	"github.com/RenatoGeh/godrive/camera"
	"github.com/RenatoGeh/godrive/data"
	"github.com/RenatoGeh/godrive/models"
	"github.com/RenatoGeh/gospn/sys"
	"os"
	"strconv"
)

func Train(t, l string, n int, filename string) {
	D, L, Sc := data.PrepareTrain(n)
	var M models.Model
	if t == "dv" {
		M = models.NewDVModel(data.ClassVar)
	} else {
		M = models.NewGensModel(data.ClassVar)
	}

	if l == "g" {
		M.LearnGenerative(D, L, Sc)
	} else if l == "d" {
		M.LearnDiscriminative(D, L, Sc)
	} else {
		M.LearnStructure(D, L, Sc)
	}

	M.Save(filename)
}

func Test(t, filename string) {
	var M models.Model
	var err error
	if t == "dv" {
		M, err = models.LoadDVModel(filename)
	} else {
		M, err = models.LoadGensModel(filename)
	}
	if err != nil {
		panic(err)
	}
	B, err := bot.New(0, M)
	if err != nil {
		panic(err)
	}
	defer B.Close()
	data.Prepare()

	B.SetTransform(camera.MakeQuantize(7))
	B.Start()
}

func Sample(t, filename string) {
	var M models.Model
	var err error
	if t == "dv" {
		M, err = models.LoadDVModel(filename)
	} else {
		M, err = models.LoadGensModel(filename)
	}
	if err != nil {
		panic(err)
	}
	const n int = 300
	D, L, _ := data.PrepareTrain(n)
	sys.StartTimer()
	M.TestAccuracy(D, L)
	d := sys.StopTimer()
	fmt.Printf("Inference took: %s\nApproximately %.2f seconds per instance.\n",
		d, d.Seconds()/float64(n))
}

func main() {
	if n := len(os.Args); n > 6 && n < 4 {
		fmt.Printf("Usage: %s r^t^s filename dv^gens g^d^s n\n", os.Args[0])
		fmt.Println("  The character ^ is used to symbolize XOR.")
		fmt.Println("    t        - test a model given by filename and run the bot")
		fmt.Println("    r        - train a model")
		fmt.Println("    s        - test model on a test dataset")
		fmt.Println("    filename - model to be loaded or saved to")
		fmt.Println("    dv^gens  - either use the Dennis-Ventura (dv) or Gens (gens) model")
		fmt.Println("  Test (t) arguments:")
		fmt.Println("  Train (r) arguments:")
		fmt.Println("    g^d^s    - either use generative (g) or discriminative (d) learning, or just structure (s)")
		fmt.Println("    n        - size of dataset to train with")
		fmt.Println("  Sample test (s) arguments:")
		return
	}
	if mode := os.Args[1]; mode == "r" {
		n, err := strconv.Atoi(os.Args[5])
		if err != nil {
			panic(err)
		}
		Train(os.Args[3], os.Args[4], n, os.Args[2])
	} else if mode == "t" {
		Test(os.Args[3], os.Args[2])
	} else if mode == "s" {
		Sample(os.Args[3], os.Args[2])
	} else {
		fmt.Println("Unrecognized option. Either r or t.")
	}
}
