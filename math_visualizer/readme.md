# Math-Physics Suite 🧮⚛️

A comprehensive web platform for mathematical computations and physics calculations with interactive visualizations.

## Features

### 🔹 Algebra Module
- Solve equations step-by-step with SymPy
- Support for linear, quadratic, cubic, trigonometric, and exponential equations
- Detailed solution explanations

### 🔹 Matrix Calculator
- 2×2 and 3×3 matrix operations
- Calculate determinants, rank, inverse, and eigenvalues
- Interactive matrix input interface

### 🔹 Trigonometry Visualizer
- Plot all trigonometric functions (sin, cos, tan, sec, csc, cot)
- Function properties and key values
- Interactive graphs from -2π to 2π

### 🔹 Calculus Tools
- Compute derivatives and integrals
- Plot original functions and their derivatives
- Support for symbolic mathematics

### 🔹 Physics Calculators
- Force calculations (F = ma)
- Wave speed on strings
- Ohm's law and electrical calculations
- Physics constants reference

### 🔹 Statistics & Data Analysis
- CSV file upload support
- Manual data entry
- Statistical measures (mean, median, std dev, etc.)
- Histogram and box plot visualizations

### 🔹 Advanced Mathematics
- Interactive 3D surface plots
- Fourier series approximations
- Riemann Zeta function visualization
- Powered by Plotly for interactivity

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone or download the project files**
   ```bash
   # Create project directory
   mkdir math-physics-suite
   cd math-physics-suite
   ```

2. **Install required packages**
   ```bash
   pip install flask sympy numpy matplotlib plotly pandas werkzeug
   ```

   Or using requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create the directory structure**
   ```
   math-physics-suite/
   ├── app.py
   ├── requirements.txt
   ├── templates/
   │   ├── base.html
   │   ├── index.html
   │   ├── algebra.html
   │   ├── matrices.html
   │   ├── trigonometry.html
   │   ├── calculus.html
   │   ├── physics.html
   │   ├── statistics.html
   │   └── advanced.html
   └── uploads/ (created automatically)
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

## Usage Examples

### Algebra
- Solve `x**2 - 4*x + 3` (quadratic equation)
- Solve `sin(x) - 0.5` (trigonometric equation)
- Solve `2*x + 5` (linear equation)

### Calculus
- Differentiate `x**3 + 2*x**2 - x + 1`
- Integrate `sin(x) + cos(x)`
- Plot `exp(x) - x**2`

### 3D Surfaces
- `x**2 + y**2` (paraboloid)
- `sin(x)*cos(y)` (wave pattern)
- `exp(-(x**2 + y**2))` (Gaussian)

### Statistics
Upload CSV files or enter data like: `12, 15, 18, 20, 22, 25, 30, 35, 40, 45`

## Technical Stack

- **Backend:** Flask (Python web framework)
- **Mathematics:** SymPy (symbolic math), NumPy (numerical computing)
- **Data Analysis:** Pandas (data manipulation)
- **Visualization:** Matplotlib (2D plots), Plotly (interactive 3D)
- **Frontend:** Bootstrap 5, HTML5, JavaScript
- **Styling:** Custom CSS with gradient backgrounds and glassmorphism

## File Structure Details

### Core Files
- `app.py` - Main Flask application with all routes and logic
- `requirements.txt` - Python package dependencies

### Templates
- `base.html` - Base template with navigation and Bootstrap styling
- `index.html` - Dashboard homepage with module cards
- `algebra.html` - Equation solver interface
- `matrices.html` - Matrix calculator with dynamic input generation
- `trigonometry.html` - Function plotter with properties
- `calculus.html` - Derivative and integral calculator
- `physics.html` - Physics calculation tools
- `statistics.html` - Data analysis with CSV upload
- `advanced.html` - 3D visualizations and advanced math

## Features Breakdown

### Interactive Elements
- ✅ Dynamic form generation (matrix inputs)
- ✅ Real-time function plotting
- ✅ Interactive 3D navigation
- ✅ CSV file upload and processing
- ✅ Step-by-step solutions
- ✅ Error handling with user-friendly messages

### Visualizations
- ✅ 2D plots with Matplotlib (base64 encoded)
- ✅ Interactive 3D plots with Plotly
- ✅ Statistical charts (histograms, box plots)
- ✅ Mathematical function graphs
- ✅ Fourier series animations

### Mathematical Capabilities
- ✅ Symbolic equation solving
- ✅ Matrix operations and linear algebra
- ✅ Calculus (derivatives, integrals, limits)
- ✅ Trigonometric function analysis
- ✅ Statistical computations
- ✅ Advanced mathematical functions

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing packages
   pip install [package-name]
   ```

2. **Port Already in Use**
   - Change port in app.py: `app.run(debug=True, port=5001)`

3. **File Upload Issues**
   - Ensure 'uploads' directory exists
   - Check file permissions

4. **Plot Display Problems**
   - Clear browser cache
   - Check matplotlib backend settings

### Performance Tips
- For large datasets, consider sampling in statistics module
- Complex 3D surfaces may take longer to render
- Use fewer Fourier terms for faster computation

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to contribute by:
- Adding new mathematical functions
- Improving visualizations
- Enhancing the user interface
- Adding more physics calculations
- Optimizing performance

---

**Enjoy exploring mathematics and physics with this comprehensive suite!** 🚀