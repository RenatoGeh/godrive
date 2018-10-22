package camera

import (
	"gocv.io/x/gocv"
	"image"
)

var (
	unitPoint = image.Point{1, 1}
)

func Binarize(src gocv.Mat, dst *gocv.Mat) {
	gocv.GaussianBlur(src, dst, unitPoint, 1, 1, gocv.BorderReflect)
	gocv.Threshold(*dst, dst, 0, 255, gocv.ThresholdOtsu)
}

func MakeQuantize(max int) func(gocv.Mat, *gocv.Mat) {
	return func(src gocv.Mat, dst *gocv.Mat) {
		k := float32(max) / 255.0
		src.CopyTo(dst)
		dst.DivideFloat(k)
	}
}

func Equalize(src gocv.Mat, dst *gocv.Mat) {
	gocv.EqualizeHist(src, dst)
}
