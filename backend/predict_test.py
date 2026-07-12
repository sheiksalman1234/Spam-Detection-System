import json, urllib.request

url = 'http://127.0.0.1:8000/predict/text'
data = json.dumps({"text": "Your bank account has been blocked. Share your OTP immediately."}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
try:
    print(urllib.request.urlopen(req, timeout=20).read().decode())
except Exception as e:
    import traceback
    print('ERROR', e)
    traceback.print_exc()
    if hasattr(e, 'read'):
        try:
            print('BODY:', e.read().decode())
        except Exception:
            pass
