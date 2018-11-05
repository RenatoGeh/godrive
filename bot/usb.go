package bot

import (
	"fmt"
	"github.com/google/gousb"
)

const (
	VendorLegoID = 0x0694
	ProductNXTID = 0x0002

	InputEndpoint  = 0x82 ^ 0x80 // = 2. Actual endpoint is 0x82, but gousb counts from 1-15.
	OutputEndpoint = 0x01

	ConfigNXT    = 1
	InterfaceNXT = 0

	blockSize = 60
)

// Port is a USB device port.
type Port struct {
	ctx *gousb.Context
	dev *gousb.Device
	cfg *gousb.Config
	itf *gousb.Interface

	out *gousb.OutEndpoint
}

func NewPort() *Port {
	ctx := gousb.NewContext()
	vid, pid := gousb.ID(VendorLegoID), gousb.ID(ProductNXTID)
	D, err := ctx.OpenDevices(func(desc *gousb.DeviceDesc) bool {
		return desc.Vendor == vid && desc.Product == pid
	})
	for _, d := range D[1:] {
		defer d.Close()
	}
	if err != nil {
		panic(err)
	}
	if len(D) == 0 {
		panic("No NXT brick found.")
	}
	dev := D[0]
	cfg, err := dev.Config(ConfigNXT)
	if err != nil {
		panic(err)
	}
	itf, err := cfg.Interface(InterfaceNXT, 0)
	if err != nil {
		panic(err)
	}
	out, err := itf.OutEndpoint(OutputEndpoint)
	if err != nil {
		panic(err)
	}
	return &Port{ctx, dev, cfg, itf, out}
}

func (p *Port) Write(b []byte) int {
	n, err := p.out.Write(b)
	if err != nil {
		fmt.Println(err)
	}
	if s := len(b); n != s {
		fmt.Printf("Lost data! Expected to send %d, sent %d.\n", s, n)
	}
	return n
}

func (p *Port) Close() {
	p.itf.Close()
	p.dev.Close()
	p.ctx.Close()
}
