"""
download_pexels.py

Alternative downloader menggunakan Pexels API (FREE!)
Lebih reliable daripada Unsplash Source API

Setup:
    1. Get FREE API key from: https://www.pexels.com/api/
    2. Set environment variable:
       Windows: $env:PEXELS_API_KEY = "your_api_key_here"
       Linux: export PEXELS_API_KEY="your_api_key_here"

Usage:
    python download_pexels.py

Output:
    data/non_faces/ - 1000 non-face images
"""

import requests
import os
import time
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def download_from_pexels(query, count, output_dir, api_key):
    """Download images from Pexels API"""
    os.makedirs(output_dir, exist_ok=True)
    
    headers = {"Authorization": api_key}
    base_url = "https://api.pexels.com/v1/search"
    
    print(f"\n📥 Downloading {count} images: '{query}'")
    
    success = 0
    page = 1
    per_page = 80  # Max per page
    
    with tqdm(total=count, desc=f"{query:12s}", unit="img") as pbar:
        while success < count:
            # Request images
            params = {
                "query": query,
                "per_page": min(per_page, count - success),
                "page": page
            }
            
            try:
                response = requests.get(base_url, headers=headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    photos = data.get("photos", [])
                    
                    if not photos:
                        print(f"\n⚠️ No more images for '{query}'")
                        break
                    
                    # Download each photo
                    for photo in photos:
                        if success >= count:
                            break
                        
                        # Get medium size image URL
                        img_url = photo["src"]["medium"]  # 350px width
                        
                        try:
                            img_response = requests.get(img_url, timeout=10)
                            if img_response.status_code == 200:
                                img = Image.open(BytesIO(img_response.content))
                                
                                # Convert to RGB if needed
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                
                                # Resize to 640x480
                                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                                
                                # Save
                                filename = f"{output_dir}/{query}_{success:04d}.jpg"
                                img.save(filename, quality=95)
                                
                                success += 1
                                pbar.update(1)
                                
                        except Exception as e:
                            print(f"\n❌ Error downloading image: {e}")
                            continue
                        
                        # Small delay
                        time.sleep(0.1)
                    
                    page += 1
                    
                elif response.status_code == 429:
                    print(f"\n⏱️ Rate limit reached, waiting 60 seconds...")
                    time.sleep(60)
                    
                else:
                    print(f"\n❌ HTTP {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"\n❌ Error: {e}")
                break
    
    print(f"✅ {success}/{count} images downloaded for '{query}'")
    return success


def main():
    # ============================================
    # 🔑 PASTE YOUR PEXELS API KEY HERE:
    # ============================================
    PEXELS_API_KEY = "REfIHEWzCyZ9DRM3Nrsf79UJa7nwQhqN2i6l0lmBXys2taiEuCRKsFdy"  # <-- Paste API key di sini (dalam quotes)
    # Example: PEXELS_API_KEY = "abcd1234efgh5678"
    
    # Check API key (prioritas: hardcoded > environment variable)
    api_key = PEXELS_API_KEY if PEXELS_API_KEY else os.getenv("PEXELS_API_KEY")
    
    if not api_key:
        print("=" * 60)
        print("❌ PEXELS API KEY NOT FOUND")
        print("=" * 60)
        print("\n📋 Setup Instructions:")
        print("1. Go to: https://www.pexels.com/api/")
        print("2. Click 'Get Started' (FREE!)")
        print("3. Copy your API key")
        print("4. Set environment variable:")
        print("   Windows PowerShell:")
        print('     $env:PEXELS_API_KEY = "your_api_key_here"')
        print("   Linux/Mac:")
        print('     export PEXELS_API_KEY="your_api_key_here"')
        print("\n5. Run this script again")
        print("=" * 60)
        return
    
    print("=" * 60)
    print("🖼️ PEXELS IMAGE DOWNLOADER")
    print("=" * 60)
    print(f"✅ API Key: {api_key[:10]}...")
    
    # Categories - TOTAL: 5000 images
    categories = {
        "building": 800,      # Gedung, arsitektur (increased from 200)
        "nature": 800,        # Alam, landscape (increased from 200)
        "car": 500,           # Kendaraan (increased from 100)
        "food": 700,          # Makanan (increased from 150)
        "animal": 500,        # Hewan (increased from 100)
        "furniture": 500,     # Furniture (increased from 100)
        "texture": 600,       # Tekstur (increased from 100)
        "plant": 300,         # Tanaman (increased from 50)
        "technology": 300     # Gadget, electronics (NEW!)
    }
    
    output_dir = "data/non_faces"
    
    print(f"📂 Output: {output_dir}")
    print(f"📊 Total: {sum(categories.values())} images")
    print("\n📋 Categories:")
    for cat, count in categories.items():
        print(f"   - {cat:12s}: {count:3d} images")
    print("=" * 60)
    
    response = input("\n🚀 Start download? (y/n): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return
    
    # Download
    total_success = 0
    start_time = time.time()
    
    for category, count in categories.items():
        success = download_from_pexels(category, count, output_dir, api_key)
        total_success += success
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"✅ Downloaded: {total_success}/{sum(categories.values())}")
    print(f"⏱️ Time: {elapsed/60:.1f} minutes")
    print(f"📂 Location: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
