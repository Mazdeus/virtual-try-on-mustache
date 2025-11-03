extends Control

@onready var back_button = $BackButton
@onready var try_now_button = $MarginContainer/VBoxContainer/BottomPanel/TryNowButton

# Step indicators
@onready var step1 = $MarginContainer/VBoxContainer/StepsPanel/MarginContainer/VBoxContainer/Step1
@onready var step2 = $MarginContainer/VBoxContainer/StepsPanel/MarginContainer/VBoxContainer/Step2
@onready var step3 = $MarginContainer/VBoxContainer/StepsPanel/MarginContainer/VBoxContainer/Step3
@onready var step4 = $MarginContainer/VBoxContainer/StepsPanel/MarginContainer/VBoxContainer/Step4
@onready var step5 = $MarginContainer/VBoxContainer/StepsPanel/MarginContainer/VBoxContainer/Step5

var current_step = 0
var steps = []
var step_timer: Timer

func _ready():
	print("=== HowToUse._ready() ===")
	print("How to Use page loaded successfully")
	
	# Collect all steps
	steps = [step1, step2, step3, step4, step5]
	
	# Connect buttons
	if back_button:
		back_button.pressed.connect(_on_back_button_pressed)
		print("✅ Back button connected")
	
	if try_now_button:
		try_now_button.pressed.connect(_on_try_now_button_pressed)
		print("✅ Try Now button connected")
	
	# Setup step animation timer
	step_timer = Timer.new()
	add_child(step_timer)
	step_timer.timeout.connect(_animate_next_step)
	step_timer.wait_time = 0.8  # Animate every 0.8 seconds
	step_timer.start()
	
	# Initially dim all steps
	for step in steps:
		_dim_step(step)
	
	# Highlight first step
	_highlight_step(steps[0])

func _animate_next_step():
	"""Animate steps sequentially for better understanding"""
	current_step += 1
	if current_step >= steps.size():
		current_step = 0
		# Dim all before restarting
		for step in steps:
			_dim_step(step)
	
	_highlight_step(steps[current_step])

func _highlight_step(step: Control):
	"""Highlight a step with animation"""
	if not step:
		return
	
	var panel = step.get_node_or_null("StepPanel")
	if panel:
		# Create scale animation
		var tween = create_tween()
		tween.set_trans(Tween.TRANS_ELASTIC)
		tween.set_ease(Tween.EASE_OUT)
		tween.tween_property(panel, "scale", Vector2(1.05, 1.05), 0.3)
		
		# Brighten color
		var color_rect = panel.get_node_or_null("ColorRect")
		if color_rect:
			color_rect.color = Color(0.2, 0.3, 0.4, 1.0)  # Highlighted blue

func _dim_step(step: Control):
	"""Dim a step"""
	if not step:
		return
	
	var panel = step.get_node_or_null("StepPanel")
	if panel:
		panel.scale = Vector2(1.0, 1.0)
		var color_rect = panel.get_node_or_null("ColorRect")
		if color_rect:
			color_rect.color = Color(0.15, 0.15, 0.15, 1.0)  # Dimmed

func _on_back_button_pressed():
	"""Navigate back to Main Menu"""
	print("Back button pressed - returning to Main Menu")
	if step_timer:
		step_timer.stop()
	get_tree().change_scene_to_file("res://Scenes/MainMenu/MainMenu.tscn")

func _on_try_now_button_pressed():
	"""Navigate to Try-On scene"""
	print("Try Now button pressed - going to Try-On")
	if step_timer:
		step_timer.stop()
	get_tree().change_scene_to_file("res://Scenes/Kumis/KumisWebcamScene.tscn")
