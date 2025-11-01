# KumisSelectionController.gd
extends Control

@onready var pilih_button = $MainContainer/ButtonContainer/PilihButton
@onready var back_button = $MainContainer/ButtonContainer/BackButton
@onready var grid_container = $MainContainer/KumisContainer/ScrollContainer/GridContainer

var selected_kumis_style: String = ""
var kumis_buttons: Dictionary = {}

# ----------------- UDP config -----------------
var server_host: String = "127.0.0.1"
var server_port: int = 8888
var kumis_ack_timeout_frames: int = 60

# ----------------- Kumis styles mapping (12 styles!) -----------------
var kumis_info := {
	"kumis_1": {
		"file": "kumis_1.png",
		"name": "Style 1",
		"preview": "res://Assets/Kumis/kumis_1.png"
	},
	"kumis_2": {
		"file": "kumis_2.png",
		"name": "Style 2",
		"preview": "res://Assets/Kumis/kumis_2.png"
	},
	"kumis_3": {
		"file": "kumis_3.png",
		"name": "Style 3",
		"preview": "res://Assets/Kumis/kumis_3.png"
	},
	"kumis_4": {
		"file": "kumis_4.png",
		"name": "Style 4",
		"preview": "res://Assets/Kumis/kumis_4.png"
	},
	"kumis_5": {
		"file": "kumis_5.png",
		"name": "Style 5",
		"preview": "res://Assets/Kumis/kumis_5.png"
	},
	"kumis_6": {
		"file": "kumis_6.png",
		"name": "Style 6",
		"preview": "res://Assets/Kumis/kumis_6.png"
	},
	"kumis_7": {
		"file": "kumis_7.png",
		"name": "Style 7",
		"preview": "res://Assets/Kumis/kumis_7.png"
	},
	"kumis_8": {
		"file": "kumis_8.png",
		"name": "Style 8",
		"preview": "res://Assets/Kumis/kumis_8.png"
	},
	"kumis_9": {
		"file": "kumis_9.png",
		"name": "Style 9",
		"preview": "res://Assets/Kumis/kumis_9.png"
	},
	"kumis_10": {
		"file": "kumis_10.png",
		"name": "Style 10",
		"preview": "res://Assets/Kumis/kumis_10.png"
	},
	"kumis_11": {
		"file": "kumis_11.png",
		"name": "Style 11",
		"preview": "res://Assets/Kumis/kumis_11.png"
	},
	"kumis_12": {
		"file": "kumis_12.png",
		"name": "Style 12",
		"preview": "res://Assets/Kumis/kumis_12.png"
	}
}

func _ready():
	print("=== KumisSelectionController._ready() ===")
	
	# Dynamically create buttons for all kumis styles
	create_kumis_buttons()
	
	# Initially disable Pilih button
	pilih_button.disabled = true
	pilih_button.text = "Pilih (Pilih kumis dulu!)"
	
	print("Kumis Selection scene initialized with %d styles" % kumis_info.size())


func create_kumis_buttons():
	"""Dynamically create button for each kumis style"""
	print("Creating buttons for %d kumis styles..." % kumis_info.size())
	
	# Sort keys by numeric order (kumis_1, kumis_2, ..., kumis_10, kumis_11, kumis_12)
	var sorted_keys = kumis_info.keys()
	sorted_keys.sort_custom(func(a, b):
		var num_a = int(a.replace("kumis_", ""))
		var num_b = int(b.replace("kumis_", ""))
		return num_a < num_b
	)
	
	for style_key in sorted_keys:
		var info = kumis_info[style_key]
		
		# Create container for button + image
		var container = VBoxContainer.new()
		container.name = style_key.capitalize() + "Container"
		container.custom_minimum_size = Vector2(120, 105)
		container.add_theme_constant_override("separation", 5)
		
		# Create preview image (TextureRect)
		var texture_rect = TextureRect.new()
		texture_rect.name = "PreviewImage"
		texture_rect.custom_minimum_size = Vector2(100, 50)
		texture_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		texture_rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
		
		# Load kumis texture
		var texture = load(info.preview)
		if texture:
			texture_rect.texture = texture
		else:
			print("⚠️ Failed to load texture: %s" % info.preview)
		
		# Add background panel to image with margin
		var panel = Panel.new()
		panel.custom_minimum_size = Vector2(110, 55)
		
		# Add margin container for padding
		var margin = MarginContainer.new()
		margin.add_theme_constant_override("margin_left", 5)
		margin.add_theme_constant_override("margin_top", 5)
		margin.add_theme_constant_override("margin_right", 5)
		margin.add_theme_constant_override("margin_bottom", 5)
		panel.add_child(margin)
		margin.add_child(texture_rect)
		container.add_child(panel)
		
		# Create button with text below image
		var button = Button.new()
		button.name = style_key.capitalize() + "Button"
		button.custom_minimum_size = Vector2(110, 30)
		button.text = info.name
		button.alignment = HORIZONTAL_ALIGNMENT_CENTER
		
		# Connect signals for interactivity
		button.pressed.connect(_on_kumis_button_pressed.bind(style_key))
		container.mouse_entered.connect(_on_kumis_button_hover.bind(style_key, true))
		container.mouse_exited.connect(_on_kumis_button_hover.bind(style_key, false))
		
		# Add button to container
		container.add_child(button)
		
		# Add container to grid
		grid_container.add_child(container)
		
		# Store button AND container reference
		kumis_buttons[style_key] = {
			"button": button,
			"container": container,
			"image": texture_rect,
			"panel": panel
		}
	
	print("✅ Created %d kumis buttons with preview images!" % kumis_buttons.size())


func _on_kumis_button_pressed(style_key: String):
	"""Handle kumis style button press"""
	print("Kumis style selected: %s" % style_key)
	
	selected_kumis_style = style_key
	
	# Update button selection visual with animation
	update_button_selection()
	
	# Enable Pilih button
	pilih_button.disabled = false
	pilih_button.text = "Pilih Kumis"
	print("Pilih button enabled")


func _on_kumis_button_hover(style_key: String, is_hovering: bool):
	"""Handle button hover for interactivity"""
	if style_key in kumis_buttons and style_key != selected_kumis_style:
		var container = kumis_buttons[style_key].container
		var panel = kumis_buttons[style_key].panel
		
		if is_hovering:
			# Scale up slightly on hover
			var tween = create_tween()
			tween.set_ease(Tween.EASE_OUT)
			tween.set_trans(Tween.TRANS_BACK)
			tween.tween_property(container, "scale", Vector2(1.05, 1.05), 0.2)
			panel.modulate = Color(0.7, 0.9, 1.0)  # Light blue highlight
		else:
			# Scale back to normal
			var tween = create_tween()
			tween.set_ease(Tween.EASE_OUT)
			tween.set_trans(Tween.TRANS_CUBIC)
			tween.tween_property(container, "scale", Vector2.ONE, 0.2)
			panel.modulate = Color.WHITE


func update_button_selection():
	"""Update visual appearance of buttons to show selection with animation"""
	# Reset all buttons with smooth transition
	for style_key in kumis_buttons.keys():
		var button = kumis_buttons[style_key].button
		var container = kumis_buttons[style_key].container
		var panel = kumis_buttons[style_key].panel
		
		# Animate color change
		var tween = create_tween()
		tween.set_ease(Tween.EASE_OUT)
		tween.set_trans(Tween.TRANS_CUBIC)
		
		if style_key == selected_kumis_style:
			# Selected: Green border/highlight with checkmark and slight scale
			tween.tween_property(panel, "modulate", Color(0.4, 1.0, 0.4), 0.3)
			container.scale = Vector2(1.08, 1.08)
			button.text = "✓ " + kumis_info[style_key].name
			button.modulate = Color(0.3, 0.9, 0.3)
		else:
			# Non-selected: White, normal scale
			tween.tween_property(panel, "modulate", Color.WHITE, 0.3)
			container.scale = Vector2.ONE
			button.text = kumis_info[style_key].name
			button.modulate = Color.WHITE
	
	if selected_kumis_style != "":
		print("Button '%s' highlighted with animation" % selected_kumis_style)


func _on_pilih_button_pressed():
	"""Handle Pilih button press"""
	print("=== Pilih button pressed ===")
	print("selected_kumis_style: %s" % selected_kumis_style)
	
	if selected_kumis_style != "":
		# Save to Global
		Global.selected_kumis_style = selected_kumis_style
		Global.kumis_enabled = true
		
		# Send command to server (without waiting for ACK)
		if kumis_info.has(selected_kumis_style):
			var kumis_file: String = kumis_info[selected_kumis_style].file
			send_kumis_to_server_fire_and_forget(kumis_file)
		
		# Go to webcam scene immediately
		get_tree().change_scene_to_file("res://Scenes/KumisNusantara/KumisWebcamScene.tscn")
	else:
		print("No kumis selected")


func _on_back_button_pressed():
	"""Return to main menu"""
	print("Back button pressed - returning to main menu")
	get_tree().change_scene_to_file("res://Scenes/MainMenu/MainMenu.tscn")


# ------------------ Networking helper ------------------
func send_kumis_to_server_fire_and_forget(kumis_filename: String):
	"""
	Send "SET_KUMIS <kumis_filename>" to UDP server without waiting for response
	"""
	var udp := PacketPeerUDP.new()
	var err = udp.connect_to_host(server_host, server_port)
	if err != OK:
		print("❌ UDP connect failed:", err)
		udp.close()
		return
	
	var message := "SET_KUMIS " + kumis_filename
	var send_result = udp.put_packet(message.to_utf8_buffer())
	if send_result != OK:
		print("❌ Failed to send SET_KUMIS:", send_result)
	else:
		print("📤 Sent to server:", message)
	
	udp.close()
