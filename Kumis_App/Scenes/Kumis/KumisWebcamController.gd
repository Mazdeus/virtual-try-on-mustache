# KumisWebcamController.gd
extends Control

@onready var webcam_display = $MainContainer/WebcamContainer/WebcamDisplay
@onready var status_label = $MainContainer/StatusContainer/StatusLabel
@onready var fps_label = $MainContainer/StatusContainer/FPSLabel
@onready var toggle_kumis_button = $MainContainer/ControlContainer/ToggleKumisButton
@onready var back_button = $MainContainer/ControlContainer/BackButton

@onready var webcam_manager = $WebcamManagerUDP

var kumis_enabled: bool = true
var frame_count: int = 0
var fps_timer: float = 0.0

func _ready():
	print("=== KumisWebcamController._ready() ===")
	
	# Enable fullscreen mode
	DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_FULLSCREEN)
	
	# Connect WebcamManager signals
	webcam_manager.frame_received.connect(_on_frame_received)
	webcam_manager.connection_changed.connect(_on_connection_changed)
	webcam_manager.error_message.connect(_on_error_message)
	
	# Connect buttons
	toggle_kumis_button.pressed.connect(_on_toggle_kumis_pressed)
	back_button.pressed.connect(_on_back_pressed)
	
	# Initialize
	status_label.text = "Connecting..."
	toggle_kumis_button.text = "Sembunyikan Kumis" if kumis_enabled else "Tampilkan Kumis"
	
	# Connect to server
	webcam_manager.connect_to_webcam_server()
	
	print("Kumis webcam scene initialized (FULLSCREEN mode)")


func _process(delta):
	# Update FPS counter
	fps_timer += delta
	if fps_timer >= 1.0:
		fps_label.text = "FPS: %d" % frame_count
		frame_count = 0
		fps_timer = 0.0
	
	# Handle fullscreen toggle (F11 key)
	if Input.is_action_just_pressed("ui_cancel"):  # ESC key
		_toggle_fullscreen()


func _toggle_fullscreen():
	"""Toggle between fullscreen and windowed mode"""
	if DisplayServer.window_get_mode() == DisplayServer.WINDOW_MODE_FULLSCREEN:
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_WINDOWED)
		print("📐 Switched to WINDOWED mode")
	else:
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_FULLSCREEN)
		print("🖥️ Switched to FULLSCREEN mode")


func _on_frame_received(texture: ImageTexture):
	"""Display received frame"""
	webcam_display.texture = texture
	frame_count += 1


func _on_connection_changed(connected: bool):
	"""Handle connection status change"""
	if connected:
		status_label.text = "✅ Connected"
		status_label.modulate = Color.GREEN
	else:
		status_label.text = "❌ Disconnected"
		status_label.modulate = Color.RED


func _on_error_message(message: String):
	"""Display error message"""
	status_label.text = "❌ " + message
	status_label.modulate = Color.RED
	print("Error:", message)


func _on_toggle_kumis_pressed():
	"""Toggle kumis overlay on/off"""
	kumis_enabled = !kumis_enabled
	
	# Send command to server
	var command = "TOGGLE_KUMIS"
	send_command_to_server(command)
	
	# Update button text
	toggle_kumis_button.text = "Sembunyikan Kumis" if kumis_enabled else "Tampilkan Kumis"
	
	print("Kumis toggled:", kumis_enabled)


func _on_back_pressed():
	"""Return to kumis selection"""
	print("Back button pressed")
	webcam_manager.disconnect_from_server()
	get_tree().change_scene_to_file("res://Scenes/Kumis/KumisSelectionScene.tscn")


func send_command_to_server(command: String):
	"""Send command to UDP server"""
	var udp = PacketPeerUDP.new()
	var err = udp.connect_to_host("127.0.0.1", 8888)
	
	if err == OK:
		udp.put_packet(command.to_utf8_buffer())
		print("📤 Sent command:", command)
	else:
		print("❌ Failed to send command:", command)
	
	udp.close()


func _notification(what):
	"""Handle scene cleanup"""
	if what == NOTIFICATION_WM_CLOSE_REQUEST or what == NOTIFICATION_PREDELETE:
		if webcam_manager:
			webcam_manager.disconnect_from_server()
