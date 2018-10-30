package models

import (
	"fmt"
	"github.com/RenatoGeh/godrive/data"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/score"
	"github.com/RenatoGeh/gospn/spn"
	"github.com/RenatoGeh/gospn/sys"
)

func printCM(M [][]int, n int) {
	fmt.Println("Confusion matrix:\n-------------------------")
	fmt.Printf("%*s  up  left right\n", 9, "")
	for i := range M {
		fmt.Printf("  %-6s:", data.DIRS[i])
		for j := range M[i] {
			fmt.Printf(" %.2f", float64(M[i][j])/float64(n))
		}
		fmt.Println("")
	}
	fmt.Println("-------------------------")
}

// Accuracy tests classification accuracy of the SPN given test dataset.
func Accuracy(S spn.SPN, T spn.Dataset, U []int, Sc map[int]*learn.Variable) {
	score := score.NewScore()
	sys.Verbose = true
	score.EvaluatePosteriorConc(T, U, S, Sc[data.ClassVarid], -1)
	sys.Verbose = false
	fmt.Println(score)
}
