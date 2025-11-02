# WebcamManagerUDP.gd - Updated for Kumis System
extends Node

signal frame_received(texture: ImageTexture)
signal connection_changed(connected: bool)
signal error_message(message: String)

var udp_client: PacketPeerUDP
var _is_connected: bool = false
var server_host: String = "127.0.0.1"
var server_port: int = 8888  # Command port
var broadcast_port: int = 9999  # Frame broadcast port

# Optimized frame assembly
var frame_buffers: Dictionary = {}
var last_completed_sequence: int = 0
var frame_timeout: float = 0.5

# Performance monitoring
var frame_count: int = 0
var packets_received: int = 0
var frames_completed: int = 0
var frames_dropped: int = 0

# Processing optimization
var max_packets_per_frame: int = 10

# Sequence handling
const SEQ_MOD_16 := 1 << 16
const SEQ_MOD_32 := 1 << 32
var seq_mod: int = 0
var max_outstanding_frames: int = 30

func _ready():
	udp_client = PacketPeerUDP.new()
	print("🥸 Kumis UDP client ready")

func connect_to_webcam_server():
	if _is_connected:
		return

	print("🔄 Connecting to kumis server...")

	# Bind to broadcast port to receive frames
	var error = udp_client.bind(broadcast_port)
	if error != OK:
		_emit_error("UDP bind failed: " + str(error))
		return

	# Send CONNECT command to server
	send_command("CONNECT")
	
	print("📤 CONNECT sent, waiting for frames...")

	# Wait briefly for first frame
	var timeout = 0
	var max_timeout = 90
	var got_frame = false

	while timeout < max_timeout and not got_frame:
		await get_tree().process_frame
		timeout += 1

		if udp_client.get_available_packet_count() > 0:
			got_frame = true
			print("✅ Connected to kumis server!")

	if got_frame:
		_is_connected = true
		connection_changed.emit(true)
		set_process(true)
		_reset_stats()
	else:
		_emit_error("Connection timeout")
		udp_client.close()

func _process(_delta):
	if not _is_connected:
		return

	# Process packets
	var processed = 0
	while processed < max_packets_per_frame and udp_client.get_available_packet_count() > 0:
		var packet = udp_client.get_packet()
		if packet.size() >= 12:
			packets_received += 1
			process_packet(packet)
		processed += 1

	# Cleanup old frames
	if packets_received > 0 and packets_received % 30 == 0:
		cleanup_old_frames()

func process_packet(packet: PackedByteArray):
	if packet.size() < 12:
		return

	var sequence_number = bytes_to_int(packet.slice(0, 4))
	var total_packets = bytes_to_int(packet.slice(4, 8))
	var packet_index = bytes_to_int(packet.slice(8, 12))
	var packet_data = packet.slice(12)

	if total_packets <= 0 or packet_index >= total_packets:
		return

	# Auto-detect sequence modulus
	if seq_mod == 0:
		if sequence_number <= 0xFFFF:
			seq_mod = SEQ_MOD_16
		else:
			seq_mod = SEQ_MOD_32

	# Reject old sequences
	if last_completed_sequence != 0:
		if sequence_number == last_completed_sequence:
			return
		if not _seq_is_newer(sequence_number, last_completed_sequence):
			return

	# Initialize buffer
	if sequence_number not in frame_buffers:
		_purge_if_exceed()
		frame_buffers[sequence_number] = {
			"total_packets": total_packets,
			"received_packets": 0,
			"data_parts": {},
			"timestamp": Time.get_ticks_msec() / 1000.0
		}

	var frame_buffer = frame_buffers[sequence_number]

	# Add packet
	if packet_index not in frame_buffer.data_parts:
		frame_buffer.data_parts[packet_index] = packet_data
		frame_buffer.received_packets += 1

		# Check completion
		if frame_buffer.received_packets == frame_buffer.total_packets:
			assemble_and_display_frame(sequence_number)

func assemble_and_display_frame(sequence_number: int):
	if sequence_number not in frame_buffers:
		return

	var frame_buffer = frame_buffers[sequence_number]
	var frame_data = PackedByteArray()

	# Assemble in order
	for i in range(frame_buffer.total_packets):
		if i in frame_buffer.data_parts:
			frame_data.append_array(frame_buffer.data_parts[i])
		else:
			frames_dropped += 1
			frame_buffers.erase(sequence_number)
			return

	# Success
	frame_buffers.erase(sequence_number)
	last_completed_sequence = sequence_number
	frames_completed += 1

	display_frame(frame_data)

	# Log stats
	if frames_completed % 60 == 0 and frames_completed > 0:
		var drop_rate = float(frames_dropped) / float(frames_completed + frames_dropped) * 100.0
		print("📊 Frames: %d, Drop rate: %.1f%%" % [frames_completed, drop_rate])

func cleanup_old_frames():
	var current_time = Time.get_ticks_msec() / 1000.0
	var to_remove: Array = []

	for seq_num in frame_buffers.keys():
		if current_time - frame_buffers[seq_num].timestamp > frame_timeout:
			to_remove.append(seq_num)
			frames_dropped += 1

	for seq_num in to_remove:
		frame_buffers.erase(seq_num)

func bytes_to_int(bytes: PackedByteArray) -> int:
	if bytes.size() != 4:
		return 0
	return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]

func display_frame(frame_data: PackedByteArray):
	var image = Image.new()
	var error = image.load_jpg_from_buffer(frame_data)

	if error == OK:
		var texture: ImageTexture = ImageTexture.create_from_image(image)
		frame_received.emit(texture)
		frame_count += 1

		if frame_count == 1:
			print("✅ Video stream active: %dx%d" % [image.get_width(), image.get_height()])
	else:
		print("❌ Frame decode error: ", error)
		frames_dropped += 1

func send_command(command: String):
	"""Send command to server (CONNECT, SET_KUMIS, TOGGLE_KUMIS, DISCONNECT)"""
	var cmd_udp = PacketPeerUDP.new()
	var err = cmd_udp.connect_to_host(server_host, server_port)
	
	if err == OK:
		cmd_udp.put_packet(command.to_utf8_buffer())
		print("📤 Command sent:", command)
	else:
		print("❌ Failed to send command:", command)
	
	cmd_udp.close()

func disconnect_from_server():
	if _is_connected:
		send_command("DISCONNECT")

	_is_connected = false
	udp_client.close()
	frame_buffers.clear()
	connection_changed.emit(false)
	set_process(false)
	_reset_stats()

func _reset_stats():
	frame_count = 0
	packets_received = 0
	frames_completed = 0
	frames_dropped = 0

func get_connection_status() -> bool:
	return _is_connected

func _emit_error(message: String):
	print("WebcamManager Error: " + message)
	error_message.emit(message)

func _notification(what):
	if what == NOTIFICATION_WM_CLOSE_REQUEST or what == NOTIFICATION_PREDELETE:
		disconnect_from_server()

# Helper functions
func _seq_is_newer(a: int, b: int) -> bool:
	if b == 0:
		return true
	var mod = seq_mod if seq_mod != 0 else SEQ_MOD_16
	var diff = (a - b + mod) % mod
	return diff != 0 and diff < int(mod / 2)

func _purge_if_exceed():
	if frame_buffers.size() <= max_outstanding_frames:
		return

	var items: Array = []
	for k in frame_buffers.keys():
		items.append({"seq": k, "ts": frame_buffers[k].timestamp})

	items.sort_custom(Callable(self, "_compare_items_by_ts"))

	var to_remove_count = frame_buffers.size() - max_outstanding_frames
	for i in range(to_remove_count):
		var seq_to_rm = items[i]["seq"]
		if seq_to_rm in frame_buffers:
			frame_buffers.erase(seq_to_rm)
			frames_dropped += 1

func _compare_items_by_ts(a, b):
	if a["ts"] < b["ts"]:
		return -1
	elif a["ts"] > b["ts"]:
		return 1
	return 0
