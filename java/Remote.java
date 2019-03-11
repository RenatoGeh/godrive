import java.io.*;
import lejos.nxt.*;
import lejos.nxt.comm.*;

public class Remote {
  private static DataOutputStream out;
  private static DataInputStream in;
  private static USBConnection usb;

  private static final byte NOOP = 0x00;
  private static final byte QUIT = 0x01;
  private static final byte UP = 0x02;
  private static final byte RIGHT = 0x03;
  private static final byte LEFT = 0x04;

  private static int speed = 200;
  private static int tSpeed = 50;
  private static byte last = NOOP;

  private static int lMS = speed;
  private static int rMS = speed;

  private static void connect() {
    System.out.println("Connecting...");
    usb = USB.waitForConnection(0, NXTConnection.RAW);
    out = usb.openDataOutputStream();
    in = usb.openDataInputStream();
  }

  private static void disconnect() throws java.io.IOException {
    System.out.println("Disconnecting...");
    out.close();
    in.close();
    USB.usbReset();
  }

  private static boolean check(byte c) {
    switch(c) {
      case UP:
        lMS = rMS = speed;
        break;
      case LEFT:
        lMS = 2*tSpeed;
        rMS = 3*tSpeed;
        break;
      case RIGHT:
        lMS = 3*tSpeed;
        rMS = 2*tSpeed;
        break;
      case QUIT:
        return true;
    }
    Motor.B.setSpeed(lMS);
    Motor.C.setSpeed(rMS);
    Motor.B.forward();
    Motor.C.forward();
    return false;
  }

  public static void main(String[] args) throws Exception {
    connect();
    while (true) {
      if (in.available() > 0) {
        byte c = in.readByte();
        if (check(c)) {
          disconnect();
          break;
        }
      }
    }
  }
}
