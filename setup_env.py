import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = [
    "pandas",
    "pandas_ta",
    "numpy",
    "matplotlib",
    "ccxt"
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"{pkg} قبلا نصب شده است ✅")
    except ImportError:
        print(f"{pkg} نصب می‌شود...")
        install(pkg)

print("تمام کتابخانه‌ها آماده است ✅")
