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
	c, err := gocv.OpenVideoCapture(id)
	if err != nil {
		return nil, err
	}
	c.Set(gocv.VideoCaptureFPS, 1)
	win := gocv.NewWindow("Camera")
	return &Camera{c, gocv.NewMat(), gocv.NewMat(), win}, nil
}

func (C *Camera) Update(T func(src gocv.Mat, dist *gocv.Mat)) {
	C.cam.Read(&C.img)
	M := gocv.NewMat()
	gocv.CvtColor(C.img, &M, gocv.ColorBGRToGray)
	gocv.Resize(M, &M, dims, 0, 0, gocv.InterpolationNearestNeighbor)
	T(M, &C.tImg)
}

func (C *Camera) Instance() map[int]int {
	buffer := C.tImg.DataPtrUint8()
	I := make(map[int]int)
	for i, p := range buffer {
		I[i] = int(p)
	}
	return I
}

func (C *Camera) Draw() {
	C.win.IMShow(C.img)
}

func (C *Camera) Close() {
	C.cam.Close()
	C.win.Close()
	C.img.Close()
}
