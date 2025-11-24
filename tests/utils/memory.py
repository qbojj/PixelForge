from amaranth_soc.memory import MemoryMap, ResourceInfo
from amaranth_soc.wishbone.bus import Interface


def get_memory_resource(mmap: MemoryMap, name: str) -> ResourceInfo:
    subvalues = name.split(".")
    subvalues = [MemoryMap.Name((*sv.split(","),)) for sv in subvalues]

    for res in mmap.all_resources():
        if len(res.path) == len(subvalues) and all(
            (res.path[i] == subvalues[i]) for i in range(len(res.path))
        ):
            return res
    raise KeyError(f"Resource {name} not found in memory map")


class DebugAccess(Interface):
    async def read_bytes(self, ctx, addr: int, width: int) -> bytes:
        """Perform read of 8-bit data."""
        assert self.granularity == 8, "Granularity must be 8 bits for read_bytes"

        ret = bytearray()
        for i in range(width):
            a = addr + i

            ctx.set(self.cyc, 1)
            ctx.set(self.stb, 1)
            ctx.set(self.we, 0)

            sel = a % (self.data_width // 8)
            adr = a // (self.data_width // 8)

            ctx.set(self.adr, adr)
            ctx.set(self.sel, 1 << sel)
            await ctx.tick().until(self.ack)
            ctx.set(self.cyc, 0)
            ctx.set(self.stb, 0)
            ret.append((ctx.get(self.dat_r) >> (sel * 8)) & 0xFF)
            await ctx.tick()

        return bytes(ret)

    async def read(self, ctx, addr: int, width: int) -> list[int]:
        """Perform a read transaction.

        Parameters
        ----------
        ctx : SimulatorProcess
            The simulation context.
        addr : int
            The address to read from.
        width : int
            The number of bytes to read.
        Returns
        -------
        list[int]
            The read data as a list of Consts, one per byte.
        """

        assert (
            addr % (self.data_width // self.granularity) == 0
        ), "Address must be aligned to data width/granularity"

        data = []
        for i in range(width):
            ctx.set(self.adr, addr + i)
            ctx.set(self.sel, ~0)
            ctx.set(self.cyc, 1)
            ctx.set(self.stb, 1)
            ctx.set(self.we, 0)
            await ctx.tick().until(self.ack)
            data.append(ctx.get(self.dat_r))
            ctx.set(self.cyc, 0)
            ctx.set(self.stb, 0)
            await ctx.tick()
        return data

    async def write_bytes(self, ctx, addr: int, data: bytes) -> None:
        """Perform writes of 8-bit data."""
        assert self.granularity == 8, "Granularity must be 8 bits for write_bytes"

        # separate data by address alignment and write them block by block
        # (each block should have different address // (data_width // 8))
        blocks: list[tuple[int, int, bytearray]] = []  # addr_block, start_idx, data
        for byte, adr in zip(data, range(addr, addr + len(data))):
            addr_block = adr // (self.data_width // 8)
            if len(blocks) == 0 or blocks[-1][0] != addr_block:
                off = adr % (self.data_width // 8)
                blocks.append((addr_block, off, bytearray([byte])))
            else:
                blocks[-1] = (
                    blocks[-1][0],
                    blocks[-1][1],
                    blocks[-1][2] + bytearray([byte]),
                )

        for addr_block, start_idx, block_data in blocks:
            # write block_data to addr_block
            ctx.set(self.cyc, 1)
            ctx.set(self.stb, 1)
            ctx.set(self.we, 1)
            ctx.set(self.adr, addr_block)
            dat_w_v = 0
            sel_v = 0
            for i, byte in enumerate(block_data, start=start_idx):
                dat_w_v |= byte << (i * 8)
                sel_v |= 1 << i
            ctx.set(self.dat_w, dat_w_v)
            ctx.set(self.sel, sel_v)
            await ctx.tick().until(self.ack)
            ctx.set(self.cyc, 0)
            ctx.set(self.stb, 0)
            await ctx.tick()

    async def write(self, ctx, addr: int, data: list[int]) -> None:
        """Perform a write transaction. on word-granularity.

        Parameters
        ----------
        ctx : SimulatorProcess
            The simulation context.
        addr : int
            The address to write to.
        data : list[int]
            The data to write as a list of Consts, one per byte.
        """

        assert (
            addr % (self.data_width // self.granularity) == 0
        ), "Address must be aligned to data width/granularity"

        for i, datum in enumerate(data):
            ctx.set(self.adr, addr // (self.data_width // self.granularity) + i)
            ctx.set(self.dat_w, datum)
            ctx.set(self.sel, ~0)
            ctx.set(self.cyc, 1)
            ctx.set(self.stb, 1)
            ctx.set(self.we, 1)
            await ctx.tick().until(self.ack)
            ctx.set(self.cyc, 0)
            ctx.set(self.stb, 0)
            await ctx.tick()
