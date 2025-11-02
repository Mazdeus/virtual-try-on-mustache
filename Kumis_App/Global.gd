# Global.gd
# Global singleton untuk menyimpan state aplikasi

extends Node

# Kumis selection state
var selected_kumis_style: String = ""
var kumis_enabled: bool = true

# Webcam connection state  
var webcam_connected: bool = false
var server_host: String = "127.0.0.1"
var server_port: int = 8888

func _ready():
	print("Global autoload initialized")
	print("Kumis Nusantara - Virtual Try-On System")
