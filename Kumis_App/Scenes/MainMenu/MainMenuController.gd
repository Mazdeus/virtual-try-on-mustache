extends Control

@onready var try_on_button = $CenterContainer/VBoxContainer/ButtonsPanel/MarginContainer/VBoxContainer/TryOnButton
@onready var how_to_use_button = $CenterContainer/VBoxContainer/ButtonsPanel/MarginContainer/VBoxContainer/HowToUseButton
@onready var about_button = $CenterContainer/VBoxContainer/ButtonsPanel/MarginContainer/VBoxContainer/AboutButton
@onready var quit_button = $CenterContainer/VBoxContainer/ButtonsPanel/MarginContainer/VBoxContainer/QuitButton

func _ready():
	print("=== MainMenu._ready() ===")
	print("Main Menu loaded successfully")
	
	# Connect button signals
	if try_on_button:
		try_on_button.pressed.connect(_on_try_on_button_pressed)
		print("✅ Try On button connected")
	else:
		print("❌ Try On button not found!")
	
	if how_to_use_button:
		how_to_use_button.pressed.connect(_on_how_to_use_button_pressed)
		print("✅ How to Use button connected")
	else:
		print("❌ How to Use button not found!")
	
	if about_button:
		about_button.pressed.connect(_on_about_button_pressed)
		print("✅ About button connected")
	else:
		print("❌ About button not found!")
	
	if quit_button:
		quit_button.pressed.connect(_on_quit_button_pressed)
		print("✅ Quit button connected")
	else:
		print("❌ Quit button not found!")

func _on_try_on_button_pressed():
	"""Navigate directly to Webcam scene"""
	print("Try On button pressed - going to Webcam")
	get_tree().change_scene_to_file("res://Scenes/Kumis/KumisWebcamScene.tscn")

func _on_how_to_use_button_pressed():
	"""Navigate to How to Use scene"""
	print("How to Use button pressed - going to Tutorial")
	get_tree().change_scene_to_file("res://Scenes/HowToUse/HowToUse.tscn")

func _on_about_button_pressed():
	"""Navigate to Contributors scene"""
	print("Contributors button pressed - going to Contributors")
	get_tree().change_scene_to_file("res://Scenes/AboutUs/AboutUs.tscn")

func _on_quit_button_pressed():
	"""Quit the application"""
	print("Quit button pressed - exiting application")
	get_tree().quit()
