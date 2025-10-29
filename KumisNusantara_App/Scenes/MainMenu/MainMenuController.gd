extends Control

@onready var try_on_button = $CenterContainer/VBoxContainer/TryOnButton
@onready var quit_button = $CenterContainer/VBoxContainer/QuitButton

func _ready():
	print("=== MainMenu._ready() ===")
	print("Main Menu loaded successfully")
	
	# Connect button signals
	if try_on_button:
		try_on_button.pressed.connect(_on_try_on_button_pressed)
		print("✅ Try On button connected")
	else:
		print("❌ Try On button not found!")
	
	if quit_button:
		quit_button.pressed.connect(_on_quit_button_pressed)
		print("✅ Quit button connected")
	else:
		print("❌ Quit button not found!")

func _on_try_on_button_pressed():
	"""Navigate to Kumis Selection scene"""
	print("Try On button pressed - going to Kumis Selection")
	get_tree().change_scene_to_file("res://Scenes/KumisNusantara/KumisSelectionScene.tscn")

func _on_quit_button_pressed():
	"""Quit the application"""
	print("Quit button pressed - exiting application")
	get_tree().quit()

