// (C) 2001-2025 Altera Corporation. All rights reserved.
// Your use of Altera Corporation's design tools, logic functions and other
// software and tools, and its AMPP partner logic functions, and any output
// files from any of the foregoing (including device programming or simulation
// files), and any associated documentation or information are expressly subject
// to the terms and conditions of the Altera Program License Subscription
// Agreement, Altera IP License Agreement, or other applicable
// license agreement, including, without limitation, that your use is for the
// sole purpose of programming logic devices manufactured by Altera and sold by
// Altera or its authorized distributors.  Please refer to the applicable
// agreement for further details.


// THIS FILE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THIS FILE OR THE USE OR OTHER DEALINGS
// IN THIS FILE.

/******************************************************************************
 *                                                                            *
 * This module removes VIP packet from a video stream and passes only RAW     *
 *  video frame packet downstream.                                            *
 *                                                                            *
 ******************************************************************************/

module soc_system_video_bridge_vip_to_raw_bridge_0 (
	// Inputs
	clk,
	reset,

	vip_data,
	vip_startofpacket,
	vip_endofpacket,
	vip_empty,
	vip_valid,

	raw_ready,

	// Bidirectional

	// Outputs
	vip_ready,

	raw_data,
	raw_startofpacket,
	raw_endofpacket,
	raw_empty,
	raw_valid
);

/*****************************************************************************
 *                           Parameter Declarations                          *
 *****************************************************************************/

parameter DW = 31;
parameter EW = 1;

/*****************************************************************************
 *                             Port Declarations                             *
 *****************************************************************************/

// Inputs
input						clk;
input						reset;

input			[DW: 0]	vip_data;
input						vip_startofpacket;
input						vip_endofpacket;
input			[EW: 0]	vip_empty;
input						vip_valid;

input						raw_ready;

// Bidirectional

// Outputs
output					vip_ready;

output reg	[DW: 0]	raw_data;
output reg				raw_startofpacket;
output reg				raw_endofpacket;
output reg	[EW: 0]	raw_empty;
output reg				raw_valid;

/*****************************************************************************
 *                           Constant Declarations                           *
 *****************************************************************************/

localparam	STATE_0_DISCARD_PACKET			= 2'h0,
				STATE_1_START_CONVERT_TO_RAW	= 2'h1,
				STATE_2_CONVERT_TO_RAW			= 2'h2;

/*****************************************************************************
 *                 Internal Wires and Registers Declarations                 *
 *****************************************************************************/

// Internal Wires
wire						incoming_vip_packet_is_frame_data;

// Internal Registers

// State Machine Registers
reg			[ 1: 0]	s_vip_to_raw;
reg			[ 1: 0]	ns_vip_to_raw;

// Integers

/*****************************************************************************
 *                         Finite State Machine(s)                           *
 *****************************************************************************/

always @(posedge clk)
begin
	if (reset)
		s_vip_to_raw <= STATE_0_DISCARD_PACKET;
	else
		s_vip_to_raw <= ns_vip_to_raw;
end

always @(*)
begin
   case (s_vip_to_raw)
	STATE_0_DISCARD_PACKET:
		begin
			if (incoming_vip_packet_is_frame_data)
				ns_vip_to_raw = STATE_1_START_CONVERT_TO_RAW;
			else
				ns_vip_to_raw = STATE_0_DISCARD_PACKET;
		end
	STATE_1_START_CONVERT_TO_RAW:
		begin
			if ((~raw_valid | raw_ready) & vip_valid)
				ns_vip_to_raw = STATE_2_CONVERT_TO_RAW;
			else
				ns_vip_to_raw = STATE_1_START_CONVERT_TO_RAW;
		end
	STATE_2_CONVERT_TO_RAW:
		begin
			if ((~raw_valid | raw_ready) &
					vip_endofpacket & vip_valid)
				ns_vip_to_raw = STATE_0_DISCARD_PACKET;
			else
				ns_vip_to_raw = STATE_2_CONVERT_TO_RAW;
		end
	default:
		begin
			ns_vip_to_raw = STATE_0_DISCARD_PACKET;
		end
	endcase
end

/*****************************************************************************
 *                             Sequential Logic                              *
 *****************************************************************************/

// Output Registers

always @(posedge clk)
begin
	if (reset)
	begin
		raw_data				<= 'h0;
		raw_startofpacket	<= 1'b0;
		raw_endofpacket	<= 1'b0;
		raw_empty			<= 1'b0;
		raw_valid			<= 1'b0;
	end
	else if ((s_vip_to_raw == STATE_1_START_CONVERT_TO_RAW) &
			(~raw_valid | raw_ready))
	begin
		raw_data				<= vip_data;
		raw_startofpacket	<= 1'b1;
		raw_endofpacket	<= vip_endofpacket;
		raw_empty			<= vip_empty;
		raw_valid			<= vip_valid;
	end
	else if ((s_vip_to_raw == STATE_2_CONVERT_TO_RAW) &
			(~raw_valid | raw_ready))
	begin
		raw_data				<= vip_data;
		raw_startofpacket	<= 1'b0;
		raw_endofpacket	<= vip_endofpacket;
		raw_empty			<= vip_empty;
		raw_valid			<= vip_valid;
	end
	else if (raw_ready)
	begin
		raw_valid			<= 1'b0;
	end
end

// Internal Registers


/*****************************************************************************
 *                            Combinational Logic                            *
 *****************************************************************************/

// Output Assignments
assign vip_ready	= (s_vip_to_raw == STATE_0_DISCARD_PACKET) ?
   						1'b1 :
						(~raw_valid | raw_ready);

// Internal Assignments
assign incoming_vip_packet_is_frame_data =
		vip_valid & vip_startofpacket & (vip_data == 'h0);

/*****************************************************************************
 *                              Internal Modules                             *
 *****************************************************************************/


endmodule
