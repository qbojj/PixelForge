# PixelForge DE1-SoC Installation Guide

This guide explains how to install and run PixelForge on an Altera DE1-SoC FPGA development board.

## Prerequisites

- DE1-SoC FPGA development board
- SD card (at least 512 MB)
- SD card reader/writer
- Linux or other Unix-like system with `dd` utility
- `disk.img` from the [PixelForge github repository](https://github.com/qbojj/PixelForge/releases)

## Installation Steps

### 1. Prepare the SD Card

Insert your SD card into a card reader connected to your Linux system. Identify the device path (e.g., `/dev/sdb`, `/dev/sdmmc0`):

```bash
lsblk
```

**WARNING:** Ensure you identify the correct device. Using the wrong device path will overwrite other drives!

The SD card should be at least 512 MB in size.

### 2. Write the System Image

Write the PixelForge system image to the SD card:

```bash
sudo dd if=images/disk.img of=<path-to-device> bs=4k
```

Replace `<path-to-device>` with your actual SD card device (e.g., `/dev/sdb`).

**Example:**
```bash
sudo dd if=images/disk.img of=/dev/sdb bs=4k status=progress
```

The `status=progress` flag shows write progress. Wait for the command to complete.

### 3. Configure Board MSEL Switches

On the DE1-SoC board, locate the **MSEL** switches (typically a 6-position DIP switch near the FPGA).

Set all MSEL switches to **0** (down position):
```
MSEL: 0 0 0 0 0 0
```

This configures the board to boot from the SD card.

### 4. Install SD Card

Once writing is complete:
1. Eject the SD card safely from your Linux system
2. Insert the SD card into the DE1-SoC SD card slot (usually on the bottom of the board)

### 5. Power Cycle the Board

1. Ensure the DE1-SoC is powered off
2. Power on the board
3. Wait for the system to boot (LED activity will indicate progress)

## Verification

After boot:
- FPGA should flash itself (switches will be connected to the LEDs above)
- The system should be ready to run PixelForge demos

If the board doesn't boot or demos don't run, check the following:
- Check MSEL switch positions (must be 00000)
- Verify the SD card write was successful (try writing again)
- Check power and connections
- Check for successful FPGA load (switches should control LEDs)

## Running Demos

You should see a login prompt, the username is `root` with no password.

You will be in `/home/root/`. The demos should be located in this directory.

Please remember to correctly connect a VGA monitor to see the output.
See [software/README.md](software/README.md) for usage instructions.

## Troubleshooting

**Board doesn't boot:**
- Verify MSEL switches are set to 00000
- Re-write the SD card image
- Try a different SD card
- Check board power connections

**Demos show no output:**
- Verify VGA monitor is connected and powered
- Check framebuffer configuration with `dump_vga_dma`
- Consult [software/README.md](software/README.md) troubleshooting section

**SD card detection issues:**
- Use a different SD card reader
- Check SD card slot for debris or damage

## Related Documentation

- [README.md](README.md) - Project overview
- [software/README.md](software/README.md) - Demo program descriptions
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture
