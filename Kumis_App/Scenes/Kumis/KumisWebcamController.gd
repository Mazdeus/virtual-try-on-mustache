# KumisWebcamController.gd
extends Control

@onready var webcam_display = $MainHBox/LeftPanel/WebcamPanel/WebcamDisplay
@onready var status_label = $MainHBox/LeftPanel/HeaderPanel/MarginContainer/HBoxContainer/StatusLabel
@onready var fps_label = $MainHBox/LeftPanel/HeaderPanel/MarginContainer/HBoxContainer/FPSLabel
@onready var toggle_kumis_button = $MainHBox/LeftPanel/ControlPanel/MarginContainer/HBoxContainer/ToggleKumisButton
@onready var back_button = $MainHBox/LeftPanel/ControlPanel/MarginContainer/HBoxContainer/BackButton
@onready var kumis_grid = $MainHBox/RightPanel/KumisScrollContainer/KumisGridContainer
@onready var color_grid = $MainHBox/RightPanel/ColorPanel/MarginContainer/VBoxContainer/ColorGrid

@onready var webcam_manager = $WebcamManagerUDP

var kumis_enabled: bool = true
var frame_count: int = 0
var fps_timer: float = 0.0
var current_kumis_index: int = 1  # Default kumis_1
var current_color: String = "BLACK"  # Default color

# List of available kumis
var kumis_list = [
	{"name": "Kumis 1", "index": 1, "image": "res://Assets/Kumis/kumis_1.png"},
	{"name": "Kumis 2", "index": 2, "image": "res://Assets/Kumis/kumis_2.png"},
	{"name": "Kumis 3", "index": 3, "image": "res://Assets/Kumis/kumis_3.png"},
	{"name": "Kumis 4", "index": 4, "image": "res://Assets/Kumis/kumis_4.png"},
	{"name": "Kumis 5", "index": 5, "image": "res://Assets/Kumis/kumis_5.png"},
	{"name": "Kumis 6", "index": 6, "image": "res://Assets/Kumis/kumis_6.png"},
	{"name": "Kumis 7", "index": 7, "image": "res://Assets/Kumis/kumis_7.png"},
	{"name": "Kumis 8", "index": 8, "image": "res://Assets/Kumis/kumis_8.png"},
	{"name": "Kumis 9", "index": 9, "image": "res://Assets/Kumis/kumis_9.png"},
	{"name": "Kumis 10", "index": 10, "image": "res://Assets/Kumis/kumis_10.png"},
	{"name": "Kumis 11", "index": 11, "image": "res://Assets/Kumis/kumis_11.png"},
	{"name": "Kumis 12", "index": 12, "image": "res://Assets/Kumis/kumis_12.png"}
]

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
	
	# Initialize kumis selection grid
	_populate_kumis_grid()
	
	# Initialize color picker
	_populate_color_picker()
	
	# Initialize
	status_label.text = "● Connecting..."
	toggle_kumis_button.text = "👁 Sembunyikan Kumis" if kumis_enabled else "👁 Tampilkan Kumis"
	
	# Connect to server and set initial kumis
	webcam_manager.connect_to_webcam_server()
	await get_tree().create_timer(0.5).timeout  # Wait for connection
	send_command_to_server("SELECT_KUMIS:%d" % current_kumis_index)
	
	print("Kumis webcam scene initialized (FULLSCREEN mode)")


func _populate_kumis_grid():
	"""Create kumis selection buttons"""
	for kumis_data in kumis_list:
		var button_container = PanelContainer.new()
		button_container.custom_minimum_size = Vector2(160, 100)
		
		var button = Button.new()
		button.custom_minimum_size = Vector2(160, 100)
		button.flat = true
		
		# Center container for proper centering
		var center_container = CenterContainer.new()
		center_container.mouse_filter = Control.MOUSE_FILTER_IGNORE
		
		# Add margin to texture to make kumis smaller and prevent overflow
		var margin = MarginContainer.new()
		margin.add_theme_constant_override("margin_left", 35)
		margin.add_theme_constant_override("margin_top", 25)
		margin.add_theme_constant_override("margin_right", 35)
		margin.add_theme_constant_override("margin_bottom", 25)
		margin.mouse_filter = Control.MOUSE_FILTER_IGNORE
		
		# Load kumis texture with proper aspect ratio
		var texture_rect = TextureRect.new()
		texture_rect.texture = load(kumis_data["image"])
		texture_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		texture_rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
		texture_rect.custom_minimum_size = Vector2(80, 40)  # Max size to prevent overflow
		texture_rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
		
		button.add_child(margin)
		margin.add_child(center_container)
		center_container.add_child(texture_rect)
		margin.set_anchors_preset(Control.PRESET_FULL_RECT)
		
		button_container.add_child(button)
		kumis_grid.add_child(button_container)
		
		# Connect button pressed
		var kumis_index = kumis_data["index"]
		button.pressed.connect(func(): _on_kumis_selected(kumis_index, button_container))
		
		# Highlight current selection
		if kumis_index == current_kumis_index:
			_highlight_kumis_button(button_container)
	
	print("Created %d kumis selection buttons" % kumis_list.size())


func _populate_color_picker():
	"""Create color selection buttons"""
	var colors = [
		{"name": "Hitam", "preset": "BLACK", "color": Color(0.1, 0.1, 0.1)},
		{"name": "Coklat", "preset": "BROWN", "color": Color(0.4, 0.25, 0.15)},
		{"name": "Pirang", "preset": "BLONDE", "color": Color(0.8, 0.65, 0.4)},
		{"name": "Merah", "preset": "RED", "color": Color(0.6, 0.2, 0.1)},
		{"name": "Abu", "preset": "GRAY", "color": Color(0.5, 0.5, 0.5)},
		{"name": "Putih", "preset": "WHITE", "color": Color(0.9, 0.9, 0.9)}
	]
	
	for color_data in colors:
		var button = Button.new()
		button.custom_minimum_size = Vector2(110, 45)
		button.text = color_data["name"]
		
		# Set button background color
		var style_box = StyleBoxFlat.new()
		style_box.bg_color = color_data["color"]
		style_box.border_width_left = 2
		style_box.border_width_right = 2
		style_box.border_width_top = 2
		style_box.border_width_bottom = 2
		style_box.border_color = Color(0.6, 0.6, 0.6, 1.0)
		style_box.corner_radius_top_left = 8
		style_box.corner_radius_top_right = 8
		style_box.corner_radius_bottom_left = 8
		style_box.corner_radius_bottom_right = 8
		button.add_theme_stylebox_override("normal", style_box)
		
		# Hover effect
		var style_box_hover = style_box.duplicate()
		style_box_hover.border_color = Color(1.0, 1.0, 1.0, 1.0)
		style_box_hover.border_width_left = 3
		style_box_hover.border_width_right = 3
		style_box_hover.border_width_top = 3
		style_box_hover.border_width_bottom = 3
		button.add_theme_stylebox_override("hover", style_box_hover)
		
		# Pressed effect
		var style_box_pressed = style_box.duplicate()
		style_box_pressed.border_color = Color(0.3, 0.6, 1.0, 1.0)
		style_box_pressed.border_width_left = 4
		style_box_pressed.border_width_right = 4
		style_box_pressed.border_width_top = 4
		style_box_pressed.border_width_bottom = 4
		button.add_theme_stylebox_override("pressed", style_box_pressed)
		
		# Text color for contrast
		if color_data["color"].v > 0.5:  # Light background
			button.add_theme_color_override("font_color", Color(0.1, 0.1, 0.1))
			button.add_theme_color_override("font_hover_color", Color(0.0, 0.0, 0.0))
		else:  # Dark background
			button.add_theme_color_override("font_color", Color(0.9, 0.9, 0.9))
			button.add_theme_color_override("font_hover_color", Color(1.0, 1.0, 1.0))
		
		color_grid.add_child(button)
		
		# Connect button
		var preset = color_data["preset"]
		button.pressed.connect(func(): _on_color_selected(preset))
	
	print("Created %d color buttons" % colors.size())


func _on_color_selected(preset: String):
	"""Handle color selection"""
	print("Selected color:", preset)
	current_color = preset
	
	# Send command to server
	send_command_to_server("COLOR:%s" % preset)


func _on_kumis_selected(kumis_index: int, button_container: PanelContainer):
	"""Handle kumis selection"""
	print("Selected kumis:", kumis_index)
	current_kumis_index = kumis_index
	
	# Send command to server to change kumis
	send_command_to_server("SELECT_KUMIS:%d" % kumis_index)
	
	# Update UI - highlight selected button
	for child in kumis_grid.get_children():
		_unhighlight_kumis_button(child)
	
	_highlight_kumis_button(button_container)


func _highlight_kumis_button(button_container: PanelContainer):
	"""Highlight selected kumis button"""
	var style_box = StyleBoxFlat.new()
	style_box.bg_color = Color(0.3, 0.6, 1.0, 0.3)  # Light blue
	style_box.border_width_left = 4
	style_box.border_width_right = 4
	style_box.border_width_top = 4
	style_box.border_width_bottom = 4
	style_box.border_color = Color(0.3, 0.6, 1.0, 1.0)  # Blue border
	style_box.corner_radius_top_left = 8
	style_box.corner_radius_top_right = 8
	style_box.corner_radius_bottom_left = 8
	style_box.corner_radius_bottom_right = 8
	button_container.add_theme_stylebox_override("panel", style_box)


func _unhighlight_kumis_button(button_container: PanelContainer):
	"""Remove highlight from kumis button"""
	var style_box = StyleBoxFlat.new()
	style_box.bg_color = Color(0.2, 0.2, 0.2, 0.5)  # Dark gray
	style_box.border_width_left = 2
	style_box.border_width_right = 2
	style_box.border_width_top = 2
	style_box.border_width_bottom = 2
	style_box.border_color = Color(0.4, 0.4, 0.4, 1.0)  # Gray border
	style_box.corner_radius_top_left = 8
	style_box.corner_radius_top_right = 8
	style_box.corner_radius_bottom_left = 8
	style_box.corner_radius_bottom_right = 8
	button_container.add_theme_stylebox_override("panel", style_box)


func _process(delta):
	# Update FPS counter
	fps_timer += delta
	if fps_timer >= 1.0:
		fps_label.text = "FPS: %d" % frame_count
		frame_count = 0
		fps_timer = 0.0
	
	# Handle fullscreen toggle (ESC key)
	if Input.is_action_just_pressed("ui_cancel"):
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
		status_label.text = "● Connected"
		status_label.modulate = Color(0.3, 1.0, 0.3, 1.0)  # Green
	else:
		status_label.text = "● Disconnected"
		status_label.modulate = Color(1.0, 0.3, 0.3, 1.0)  # Red


func _on_error_message(message: String):
	"""Display error message"""
	status_label.text = "● " + message
	status_label.modulate = Color(1.0, 0.3, 0.3, 1.0)  # Red
	print("Error:", message)


func _on_toggle_kumis_pressed():
	"""Toggle kumis overlay on/off"""
	kumis_enabled = !kumis_enabled
	
	# Send command to server
	var command = "TOGGLE_KUMIS"
	send_command_to_server(command)
	
	# Update button text
	toggle_kumis_button.text = "👁 Sembunyikan Kumis" if kumis_enabled else "👁 Tampilkan Kumis"
	
	print("Kumis toggled:", kumis_enabled)


func _on_back_pressed():
	"""Return to main menu"""
	print("Back button pressed")
	webcam_manager.disconnect_from_server()
	get_tree().change_scene_to_file("res://Scenes/MainMenu/MainMenu.tscn")


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
