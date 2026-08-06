# Natural Ventilation Design Explorer

The Natural Ventilation Design Explorer is an interactive educational web app for building physics. Students adjust window parameters to instantly visualize how wind and thermal buoyancy impact a room's air exchange rate through real-time models and heatmaps.

## 🌟 Features
* **Interactive Controls:** Adjust window type (side-hung, top-hung, sliding, pivot, bottom-hung), size, opening angle, and horizontal/vertical placement on the wall.
* **Real-Time Visualization:** Generates a dynamic visual representation of the window frame, glass reflections, and shadows over a brick wall based on the user's current parameters.
* **Scientific Analytics:** Calculates and displays contour plots for both "Size Effect" and "Position Effect" to help users discover optimal ventilation strategies.
* **High Performance:** Powered by NumPy backend with caching.

## 🔬 Scientific Background
This application acts as a visual and mathematical bridge for building thermodynamics. It calculates the Air Exchange Rate (ACH) by combining two primary driving forces:
1. **Thermal Buoyancy (Stack Effect):** Driven by indoor/outdoor temperature differentials and the effective height of the window opening.
2. **Wind-Driven Flow:** Driven by dynamic wind pressure, wind speed, and the specific efficiency of the chosen window type.

The core equations are derived from fundamental fluid dynamics (such as Torricelli's Law) and utilize empirical coefficients characteristic of applied research on natural and hybrid ventilation systems in Nordic climates. The modeling assumptions and parameters take inspiration from studies conducted by SINTEF and prominent researchers such as Kleiven and Hestnes.



## 🚀 Installation and Setup

### 1. Clone the repository
```bash
git clone [https://github.com/timothee-allio/Ventilation_app.git](https://github.com/timothee-allio/Ventilation_app.git)
cd Ventilation_app
```

### 2. Create and activate a virtual environment

**On Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate

```


**On macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate

```

### 3. Install the dependencies

Make sure your virtual environment is active, then install the required Python libraries:

```bash
pip install dash dash-bootstrap-components plotly numpy pillow

```

### 4. Run the application

```bash
python Ventilation_app.py

```

Once the server starts, open your web browser and navigate to `http://127.0.0.1:8050/` to explore the dashboard.

## 👨‍💻 Author

**Timothée Allio**

Developed as an interactive E-learning tool to introduce building physics and air exchange dynamics.
