"""
download_unsplash.py

Script untuk download gambar non-face dari Unsplash Source API
Gratis, tidak perlu API key, high-quality images!

Usage:
    python download_unsplash.py

Output:
    data/non_faces/ - 1000 gambar non-face dari berbagai kategori
"""

import requests
import os
import time
import argparse
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def download_unsplash_images(query, count, output_dir, resolution="640x480"):
    """
    Download gambar dari Unsplash Source API
    
    Args:
        query (str): Search query (category)
        count (int): Jumlah gambar yang akan didownload
        output_dir (str): Output directory
        resolution (str): Image resolution (default: 640x480)
    
    Returns:
        int: Jumlah gambar yang berhasil didownload
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📥 Downloading {count} images: '{query}' ({resolution})")
    
    success = 0
    retries = 0
    max_retries = 3
    
    with tqdm(total=count, desc=f"{query:12s}", unit="img") as pbar:
        i = 0
        while i < count:
            # Unsplash Source API - returns random image each time
            # Format: https://source.unsplash.com/{WIDTH}x{HEIGHT}/?{KEYWORD}
            url = f"https://source.unsplash.com/{resolution}/?{query}"
            
            try:
                response = requests.get(url, timeout=15, allow_redirects=True)
                
                if response.status_code == 200:
                    # Load image
                    img = Image.open(BytesIO(response.content))
                    
                    # Convert RGBA to RGB if needed (untuk save as JPEG)
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    elif img.mode not in ('RGB', 'L'):
                        img = img.convert('RGB')
                    
                    # Save with unique filename
                    filename = f"{output_dir}/{query}_{i:04d}.jpg"
                    img.save(filename, quality=95, optimize=True)
                    
                    success += 1
                    i += 1
                    retries = 0  # Reset retries on success
                    pbar.update(1)
                    
                else:
                    print(f"\n⚠️ HTTP {response.status_code} for {query}")
                    retries += 1
                    if retries >= max_retries:
                        print(f"❌ Max retries reached for {query}, skipping...")
                        i += 1
                        retries = 0
                        pbar.update(1)
                    
            except requests.exceptions.Timeout:
                print(f"\n⏱️ Timeout for {query}, retrying...")
                retries += 1
                if retries >= max_retries:
                    i += 1
                    retries = 0
                    pbar.update(1)
                    
            except Exception as e:
                print(f"\n❌ Error downloading {query}_{i}: {e}")
                retries += 1
                if retries >= max_retries:
                    i += 1
                    retries = 0
                    pbar.update(1)
            
            # Longer delay to avoid rate limiting (increased from 0.5s to 3s)
            time.sleep(3.0)
    
    print(f"✅ {success}/{count} images successfully downloaded for '{query}'")
    return success


def main():
    parser = argparse.ArgumentParser(description="Download non-face images from Unsplash")
    parser.add_argument("--output", type=str, default="data/non_faces",
                        help="Output directory (default: data/non_faces)")
    parser.add_argument("--total", type=int, default=1000,
                        help="Total images to download (default: 1000)")
    parser.add_argument("--resolution", type=str, default="640x480",
                        help="Image resolution (default: 640x480)")
    
    args = parser.parse_args()
    
    # Categories dan distribusi gambar
    # Total: 1000 gambar (sesuai default, bisa diubah dengan --total)
    categories = {
        "building": 200,      # Gedung, arsitektur, urban
        "nature": 200,        # Alam, landscape, gunung, pantai
        "car": 100,           # Kendaraan (mobil, motor, truck)
        "food": 150,          # Makanan dan minuman
        "animal": 100,        # Hewan (kucing, anjing, burung, dll)
        "furniture": 100,     # Furniture (kursi, meja, sofa)
        "texture": 100,       # Tekstur (kayu, batu, kain, dinding)
        "plant": 50           # Tanaman (non-landscape)
    }
    
    # Adjust proportions jika user specify total berbeda
    if args.total != 1000:
        total_ratio = args.total / 1000
        categories = {k: max(1, int(v * total_ratio)) for k, v in categories.items()}
    
    print("=" * 60)
    print("🥸 UNSPLASH NON-FACE IMAGE DOWNLOADER")
    print("=" * 60)
    print(f"📂 Output directory: {args.output}")
    print(f"📊 Total images: {sum(categories.values())}")
    print(f"📐 Resolution: {args.resolution}")
    print("\n📋 Categories:")
    for cat, count in categories.items():
        print(f"   - {cat:12s}: {count:3d} images")
    print("=" * 60)
    
    # Confirm
    response = input("\n🚀 Start download? (y/n): ")
    if response.lower() != 'y':
        print("❌ Download cancelled")
        return
    
    # Start downloading
    total_success = 0
    start_time = time.time()
    
    for category, count in categories.items():
        success = download_unsplash_images(category, count, args.output, args.resolution)
        total_success += success
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"✅ Successfully downloaded: {total_success}/{sum(categories.values())} images")
    print(f"⏱️ Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"📂 Saved to: {args.output}")
    print(f"📊 Average: {elapsed/total_success:.2f} seconds per image")
    print("=" * 60)
    
    # Next steps
    print("\n📋 NEXT STEPS:")
    print("1. Check images: ls data/non_faces/*.jpg | wc -l")
    print("2. Collect face images (500+) from LFW or webcam")
    print("3. Train model: python app.py train --pos_dir data/faces --neg_dir data/non_faces")
    print("\n🥸 Happy training!")


if __name__ == "__main__":
    main()
