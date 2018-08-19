package data

import (
	"testing"
)

func TestPrepare(t *testing.T) {
	M := []int{50, 75, 100}
	for _, m := range M {
		for _, n := range M {
			D, L, E, U := pullData("train.npy", "test.npy", m, n)
			if len(D) != len(L) {
				t.Errorf("Dataset and labels slice lengths do not match (%d, %d).", len(D), len(L))
			}
			if len(D) != m {
				t.Errorf("Dataset size does not match, got: %d, want: %d.", len(D), m)
			}
			if len(E) != len(U) {
				t.Errorf("Dataset and labels slice lengths do not match (%d, %d).", len(E), len(U))
			}
			if len(E) != n {
				t.Errorf("Dataset size does not match, got: %d, want: %d.", len(E), n)
			}
			C, K := make([]int, 3), make([]int, 3)
			for _, l := range L {
				C[l]++
			}
			for _, u := range U {
				K[u]++
			}
			for i := 0; i < 3; i++ {
				if r, l := C[i], (m/3)+1; r > l {
					t.Errorf("Dataset is unbalanced, got: %d, want: <= %d.", r, l)
				}
				if r, l := K[i], (n/3)+1; r > l {
					t.Errorf("Dataset is unbalanced, got: %d, want: <= %d.", r, l)
				}
			}
		}
	}
}
