"""
telegram_bot.py - Asynchronous Telegram Bot API photo alerts.

Uses standard Python urllib (no external dependencies) to send screenshots
with HUD overlays in a background thread to prevent camera lag.
"""

import urllib.request
import urllib.error
import threading
import uuid

def send_telegram_photo(token: str, chat_id: str, photo_bytes: bytes, caption: str):
    """Sends a photo to a Telegram chat using standard library urllib."""
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    boundary = f"Boundary-{uuid.uuid4().hex}"
    
    # Construct multipart/form-data body
    body = []
    
    # Add chat_id field
    body.append(f"--{boundary}".encode('utf-8'))
    body.append(f'Content-Disposition: form-data; name="chat_id"'.encode('utf-8'))
    body.append(b'')
    body.append(chat_id.encode('utf-8'))
    
    # Add caption field
    body.append(f"--{boundary}".encode('utf-8'))
    body.append(f'Content-Disposition: form-data; name="caption"'.encode('utf-8'))
    body.append(b'')
    body.append(caption.encode('utf-8'))
    
    # Add photo file field
    body.append(f"--{boundary}".encode('utf-8'))
    body.append(f'Content-Disposition: form-data; name="photo"; filename="screenshot.jpg"'.encode('utf-8'))
    body.append(b'Content-Type: image/jpeg')
    body.append(b'')
    body.append(photo_bytes)
    
    # End boundary
    body.append(f"--{boundary}--".encode('utf-8'))
    body.append(b'')
    
    data = b'\r\n'.join(body)
    
    req = urllib.request.Request(url, data=data)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    req.add_header('Content-Length', str(len(data)))
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as e:
        print(f"[Telegram] HTTP Error sending photo: {e.code} {e.reason}")
        try:
            print(f"[Telegram] Error response: {e.read().decode('utf-8')}")
        except Exception:
            pass
    except Exception as e:
        print(f"[Telegram] Error sending photo: {e}")

def send_telegram_photo_async(token: str, chat_id: str, photo_bytes: bytes, caption: str):
    """Dispatches photo sending to a background thread to prevent UI freezing."""
    if not token or not chat_id:
        return
    thread = threading.Thread(
        target=send_telegram_photo,
        args=(token, chat_id, photo_bytes, caption),
        daemon=True
    )
    thread.start()
