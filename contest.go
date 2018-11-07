package main

import (
	"fmt"
	"github.com/RenatoGeh/godrive/bot"
)

// ConTest - Connection test

func input(usb *bot.Port) {
	const prompt = "Connection test commands:\n" +
		"  0 - up\n" +
		"  1 - left\n" +
		"  2 - right\n" +
		"  3 - quit\n"
	for {
		var c int
		fmt.Print(prompt)
		fmt.Scanf("%d", &c)
		usb.Write([]byte{byte(c)})
		if c == 3 {
			return
		}
	}
}

func ConStart() {
	usb := bot.NewPort()
	input(usb)
	usb.Close()
}
