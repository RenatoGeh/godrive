package bot

import (
	"bufio"
	"fmt"
	"github.com/RenatoGeh/godrive/camera"
	"github.com/RenatoGeh/godrive/models"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"os"
	"sync"
)

var (
	commands = [3]string{"UP", "LEFT", "RIGHT"}
	blue     = color.RGBA{0, 0, 255, 0}
	textPt   = image.Point{10, 10}
)

type instance struct {
	I  map[int]int // The instance
	lP []float64   // Last probabilities
	P  []float64   // Probabilities
	C  int         // Predicted class
	L  *sync.Mutex // Lock
}

type Bot struct {
	cam  *camera.Camera
	mdl  models.Model
	t    func(src gocv.Mat, dst *gocv.Mat)
	inst instance
	quit bool
}

func New(id int, M models.Model) (*Bot, error) {
	C, err := camera.New(id)
	if err != nil {
		return nil, err
	}
	return &Bot{C, M, nil, instance{
		nil,
		[]float64{0, 0, 0},
		[]float64{0, 0, 0},
		0,
		&sync.Mutex{},
	}, false}, nil
}

func (B *Bot) SetTransform(T func(src gocv.Mat, dst *gocv.Mat)) {
	B.t = T
}

func (B *Bot) DoCamera() {
	for {
		B.cam.Update(B.t)
		B.cam.Draw(func(src gocv.Mat, dst *gocv.Mat) {
			src.CopyTo(dst)
			text := fmt.Sprintf("Pr(X=UP) = %.3f\nPr(X=LEFT) = %.3f\nPr(X=RIGHT) = %.3f\nPredicted: %s",
				B.inst.P[0], B.inst.P[1], B.inst.P[2], commands[B.inst.C])
			gocv.PutText(dst, text, textPt, gocv.FontHersheySimplex, 0.5, blue, 1)
		})
		B.inst.L.Lock()
		B.inst.I = B.cam.Instance()
		B.inst.L.Unlock()
		B.inst.P = B.inst.lP
		if B.quit {
			return
		}
	}
}

func (B *Bot) DoInference() {
	for {
		B.inst.L.Lock()
		I := make(map[int]int)
		for k, v := range B.inst.I {
			I[k] = v
		}
		B.inst.L.Unlock()
		c, P := B.mdl.Infer(I)
		B.inst.lP = P
		B.inst.C = c
		if B.quit {
			return
		}
	}
}

func (B *Bot) Start() {
	var wait sync.WaitGroup

	wait.Add(2)

	fmt.Println("Started bot.")
	go func() {
		fmt.Println("Press 'q' and enter to end the bot's life. :(")
		for {
			in := bufio.NewReaderSize(os.Stdin, 1)
			c, _ := in.ReadByte()
			if c == 113 { // q to quit
				B.quit = true
				return
			}
		}
	}()

	go func() {
		go B.DoCamera()
		go B.DoInference()
	}()

	wait.Wait()
	fmt.Println("Bye!")
}

func (B *Bot) Close() {
	B.cam.Close()
}
