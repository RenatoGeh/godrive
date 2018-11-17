package camera

import (
	"fmt"
	"gocv.io/x/gocv"
)

type WriterType int

const (
	WriterTypeWindow   WriterType = 0
	WriterTypeRecorder WriterType = 1
)

type CameraWriter interface {
	Write(M gocv.Mat)
	Close()
}

type Recorder struct {
	vc *gocv.VideoWriter
}

func NewRecorder(w, h int) *Recorder {
	vc, err := gocv.VideoWriterFile("out.avi", "MJPG", 10, w, h, true)
	if err != nil {
		panic(err)
	}
	return &Recorder{vc}
}

func (r *Recorder) Write(M gocv.Mat) {
	if err := r.vc.Write(M); err != nil {
		fmt.Println(err)
	}
}

func (r *Recorder) Close() {
	if err := r.vc.Close(); err != nil {
		fmt.Println(err)
	}
}

type Window struct {
	win *gocv.Window
}

func NewWindow() *Window {
	return &Window{gocv.NewWindow("Camera")}
}

func (w *Window) Write(M gocv.Mat) {
	w.win.IMShow(M)
	w.win.WaitKey(1)
}

func (w *Window) Close() {
	w.win.Close()
}
