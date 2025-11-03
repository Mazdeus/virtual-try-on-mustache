extends Control

@onready var back_button = $BackButton

func _ready():
	print("=== Contributors._ready() ===")
	print("Contributors page loaded successfully")
	
	# Connect back button
	if back_button:
		back_button.pressed.connect(_on_back_button_pressed)
		print("✅ Back button connected")
	else:
		print("❌ Back button not found!")

func _on_back_button_pressed():
	"""Navigate back to Main Menu"""
	print("Back button pressed - returning to Main Menu")
	get_tree().change_scene_to_file("res://Scenes/MainMenu/MainMenu.tscn")
