package main

import (
	"fmt"
	"github.com/RenatoGeh/godrive/bot"
	"github.com/RenatoGeh/godrive/camera"
	"github.com/RenatoGeh/godrive/data"
	"github.com/RenatoGeh/godrive/models"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/spn"
	"github.com/RenatoGeh/gospn/sys"
	"os"
	"strconv"
)

func Train(t, l string, n int, filename string, m int, tname string) {
	var D spn.Dataset
	var L []int
	var Sc map[int]*learn.Variable

	if m > 0 {
		D, L, Sc = data.PrepareTrain(n)
	} else {
		D, L, Sc = data.PrepareFrom(n, m, tname)
	}

	var M models.Model
	if t == "dv" {
		M = models.NewDVModel(data.ClassVar)
	} else {
		M = models.NewGensModel(data.ClassVar)
	}

	sys.StartTimer()
	if l == "g" {
		M.LearnGenerative(D, L, Sc)
	} else if l == "d" {
		M.LearnDiscriminative(D, L, Sc)
	} else {
		M.LearnStructure(D, L, Sc)
	}
	d := sys.StopTimer()
	fmt.Printf("Training took: %s\n", d)

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

func Sample(t, filename string, m int, tname string) {
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
	const n int = 500
	var D spn.Dataset
	var L []int
	if m > 0 {
		D, L, _ = data.PrepareTrain(n)
	} else {
		D, L, _ = data.PrepareFrom(n, m, tname)
	}
	sys.StartTimer()
	M.TestAccuracy(D, L)
	d := sys.StopTimer()
	fmt.Printf("Inference took: %s\nApproximately %.2f seconds per instance.\n",
		d, d.Seconds()/float64(n))
}

func main() {
	if n := len(os.Args); n > 8 || n < 4 {
		fmt.Printf("Usage: %s r^t^s filename dv^gens g^d^s n [m] [tname]\n", os.Args[0])
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
		fmt.Println("    m        - max pixel value")
		fmt.Println("    tname    - training set name")
		fmt.Println("  Sample test (s) arguments:")
		fmt.Println("    m        - max pixel value")
		fmt.Println("    tname    - training set name")
		return
	}
	if mode := os.Args[1]; mode == "r" {
		n, err := strconv.Atoi(os.Args[5])
		if err != nil {
			panic(err)
		}
		if len(os.Args) == 8 {
			m, err := strconv.Atoi(os.Args[6])
			if err != nil {
				panic(err)
			}
			Train(os.Args[3], os.Args[4], n, os.Args[2], m, os.Args[7])
		} else {
			Train(os.Args[3], os.Args[4], n, os.Args[2], -1, "")
		}
	} else if mode == "t" {
		Test(os.Args[3], os.Args[2])
	} else if mode == "s" {
		if len(os.Args) == 6 {
			m, err := strconv.Atoi(os.Args[4])
			if err != nil {
				panic(err)
			}
			Sample(os.Args[3], os.Args[2], m, os.Args[5])
		} else {
			Sample(os.Args[3], os.Args[2], -1, "")
		}
	} else {
		fmt.Println("Unrecognized option. Either r or t.")
	}
}
