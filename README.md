# Spatiotemporal Routing Algorithm — Shiny App

A web app that computes routes optimized for changing weather conditions (rain, heat, wind, humidity) using OSMnx/NetworkX and visualizes them with ipyleaflet. Built with Shiny for Python.

---

## Quick start

### Create a Virtual environment (Optional)

Creating a virtual environment isolates your project dependencies from your system Python.

**Windows (Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure data path:

Create a `.env` file in the project root with the absolute path to the sample data folder:

Windows example:
```
BASE_DATA_DIR=C:\Users\YourUser\Desktop\Spatiotemporal-Routing-Algorithm\20250706\t00z\outputs\
```

Linux example:
```
BASE_DATA_DIR=/home/youruser/Desktop/Spatiotemporal-Routing-Algorithm/20250706/t00z/outputs/
```

If the `.env` approach fails, you can hardcode the path in `app.py`:
```py
BASE_DATA_DIR = r"C:\full\path\to\20250706\t00z\outputs\"
```

Required NetCDF files inside that `outputs` folder:
- RAIN.nc
- T2.nc
- WSPD10.nc
- WDIR10.nc
- RH2.nc

3. Run the app:
```bash
shiny run --launch-browser app.py
```
Default address: `http://localhost:8000`

To change port:
```bash
shiny run --launch-browser --port 3000 app.py
```

## Notes & prerequisites

- Tested on Windows with Python 3.11.4 and python 3.9.13 and Linux with Python 3.9.21. Project should work on other platforms, but geospatial packages may require extra setup.
- Use absolute paths for `BASE_DATA_DIR`.