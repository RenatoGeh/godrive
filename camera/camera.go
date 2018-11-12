package camera

import (
	"github.com/RenatoGeh/godrive/data"
	"gocv.io/x/gocv"
	"image"
)

var (
	dims = image.Point{data.Width, data.Height}
)

type Camera struct {
	// Camera
	cam *gocv.VideoCapture
	// Original image matrix
	img gocv.Mat
	// Transformed image matrix
	tImg gocv.Mat
	// Window frame
	win *gocv.Window
}

// New creates a new Camera. Parameter id is the USB device ID to be used.
func New(id int) (*Camera, error) {
	c, err := gocv.OpenVideoCapture(0)
	if err != nil {
		return nil, err
	}
	win := gocv.NewWindow("Camera")
	return &Camera{c, gocv.NewMat(), gocv.NewMat(), win}, nil
}

// Update applies a transformation to the original video frame. If T is nil, then only apply
// grayscale and resizing.
func (C *Camera) Update(T func(src gocv.Mat, dst *gocv.Mat)) {
	C.cam.Read(&C.img)
	M := gocv.NewMat()
	gocv.CvtColor(C.img, &M, gocv.ColorBGRToGray)
	gocv.Resize(M, &M, dims, 0, 0, gocv.InterpolationLinear)
	if T != nil {
		T(M, &C.tImg)
	} else {
		C.tImg = M
	}
}

// Instance converts a gocv.Mat to an spn.VarSet.
func (C *Camera) Instance() map[int]int {
	I := make(map[int]int)
	for x := 0; x < data.Height; x++ {
		for y := 0; y < data.Width; y++ {
			q := y + x*data.Width
			v := int(C.tImg.GetUCharAt(x, y))
			I[q] = v
		}
	}
	return I
}

// Draw draws the original video frame to a window. If D is not nil, then apply some drawing
// function to it first.
func (C *Camera) Draw(D func(src gocv.Mat, dst *gocv.Mat)) {
	if D == nil {
		C.win.IMShow(C.img)
	} else {
		out := gocv.NewMat()
		D(C.img, &out)
		C.win.IMShow(out)
		C.win.WaitKey(1)
	}
}

// Close closes all buffers.
func (C *Camera) Close() {
	C.cam.Close()
	C.win.Close()
	C.img.Close()
}
