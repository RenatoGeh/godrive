package bot

import (
	"fmt"
	"github.com/RenatoGeh/godrive/camera"
	"github.com/RenatoGeh/godrive/models"
	"github.com/RenatoGeh/gospn/sys"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"sync"
	"time"
)

const (
	QUIT byte = 0x03
	NOOP byte = 0x04
)

var (
	commands = [3]string{"UP", "LEFT", "RIGHT"}
	blue     = color.RGBA{0, 0, 255, 0}
	red      = color.RGBA{255, 0, 0, 0}
	upPt     = image.Point{25, 25}
	leftPt   = image.Point{25, 50}
	rightPt  = image.Point{25, 75}
	predPt   = image.Point{25, 100}
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
	usb  *Port
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
	//P := NewPort()
	var P *Port
	return &Bot{C, P, M, nil, instance{
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
			up := fmt.Sprintf("Pr(X=UP) = %.3f", B.inst.P[0])
			left := fmt.Sprintf("Pr(X=RIGHT) = %.3f", B.inst.P[1])
			right := fmt.Sprintf("Pr(X=LEFT) = %.3f", B.inst.P[2])
			pred := fmt.Sprintf("Predicted: %s", commands[B.inst.C])
			gocv.PutText(dst, up, upPt, gocv.FontHersheySimplex, 0.5, blue, 2)
			gocv.PutText(dst, left, leftPt, gocv.FontHersheySimplex, 0.5, blue, 2)
			gocv.PutText(dst, right, rightPt, gocv.FontHersheySimplex, 0.5, blue, 2)
			gocv.PutText(dst, pred, predPt, gocv.FontHersheySimplex, 0.5, red, 2)
		})
		B.inst.I = B.cam.Instance()
		B.inst.P = B.inst.lP
		time.Sleep(50 * time.Millisecond)
		if B.quit {
			return
		}
	}
}

func (B *Bot) DoInference() {
	for {
		if B.inst.I == nil {
			continue
		}
		I := B.inst.I
		sys.StartTimer()
		c, P := B.mdl.Infer(I)
		B.inst.lP = P
		B.inst.C = c
		d := sys.StopTimer()
		fmt.Printf("Predicted: %d and took %s.\n", c, d)
		//B.usb.Write([]byte{byte(c)})
		if B.quit {
			//B.usb.Write([]byte{QUIT})
			return
		}
	}
}

func (B *Bot) Start() {
	fmt.Println("Started bot.")
	go func() {
		fmt.Println("Press '0' and enter to end the bot's life. :(")
		var c int
		for {
			fmt.Scanf("%d", &c)
			if c == 0 {
				B.quit = true
				return
			}
		}
	}()

	go B.DoInference()
	B.DoCamera()

	fmt.Println("Preparing to shutdown...")
	time.Sleep(2 * time.Second) // Wait for everything to end.
	fmt.Println("Bye!")
}

func (B *Bot) Close() {
	B.cam.Close()
}
