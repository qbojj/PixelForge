#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>

#define RSTMGR_BASE      0xFFD05000
#define FPGAPORTRST_OFF  0x44
#define SYSMGR_BASE 0xFFD08000
#define STATICCTRL  0x80
#define PAGE_SIZE        4096

int main() {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return 1;
    void *sysmgr_base = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, SYSMGR_BASE);
    volatile uint32_t *staticctrl = (uint32_t *)(sysmgr_base + STATICCTRL);

    printf("System Manager StaticCtrl: 0x%08X\n", *staticctrl);

    // Map the Reset Manager
    void *virtual_base = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, RSTMGR_BASE);

    // Pointer to the specific register 0xFFD05044
    volatile uint32_t *fpgaportrst = (uint32_t *)(virtual_base + FPGAPORTRST_OFF);

    printf("Current Reset State: 0x%08X\n", *fpgaportrst);

    // Release all 6 ports from reset
    *fpgaportrst = 0x3F;

    printf("New Reset State:     0x%08X\n", *fpgaportrst);
    printf("FPGA-to-SDRAM bridge ports released from reset.\n");

    munmap(virtual_base, PAGE_SIZE);
    munmap(sysmgr_base, PAGE_SIZE);
    close(fd);
    return 0;
}
