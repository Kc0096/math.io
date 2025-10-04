from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
import base64
import io
import os
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def create_plot_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

# ALGEBRA MODULE (detailed step-by-step)
@app.route('/algebra', methods=['GET', 'POST'])
def algebra():
    if request.method == 'POST':
        try:
            equation = request.form['equation']
            variable = request.form.get('variable', 'x')

            # Parse equation
            x = sp.Symbol(variable)
            expr = sp.sympify(equation)

            # Ensure equation is in "expr = 0" form
            eq = sp.Eq(expr, 0)
            steps = []
            steps.append(f"🔹 Original Equation: {sp.pretty(eq)}")

            # Expand
            expanded = sp.expand(expr)
            if expanded != expr:
                steps.append(f"Step 1: Expand → {sp.pretty(expanded)} = 0")

            # Simplify
            simplified = sp.simplify(expanded)
            if simplified != expanded:
                steps.append(f"Step 2: Simplify → {sp.pretty(simplified)} = 0")

            # Factor
            factored = sp.factor(simplified)
            if factored != simplified:
                steps.append(f"Step 3: Factor → {sp.pretty(factored)} = 0")

            # Detect degree
            degree = sp.degree(simplified, x)
            steps.append(f"Step 4: Degree of equation = {degree}")

            # Solve equation
            solutions = sp.solve(eq, x)

            if degree == 1:
                # Linear equation → Show step-by-step isolation
                steps.append("📘 Solving as a linear equation:")

                # Example: 2*x + 4 - 6 → 2*x - 2 = 0
                lhs = simplified
                const_terms = lhs.as_independent(x, as_Add=True)[0]
                coeff_x = lhs.coeff(x)

                steps.append(f"   1. Rearrange terms: {coeff_x}*{variable} + {const_terms} = 0")
                if const_terms != 0:
                    steps.append(f"   2. Move constant to RHS: {coeff_x}*{variable} = {-const_terms}")
                if coeff_x != 1:
                    steps.append(f"   3. Divide both sides by {coeff_x}: {variable} = {(-const_terms)}/{coeff_x}")

                steps.append(f"   ✅ Final Solution: {variable} = {solutions[0]}")

            elif degree == 2:
                steps.append("📘 Quadratic detected → Using factorization/quadratic formula.")
                for i, sol in enumerate(solutions):
                    steps.append(f"   ➝ Solution {i+1}: {variable} = {sp.pretty(sol)}")

            else:
                steps.append("📘 Higher order equation detected → Direct symbolic solve.")
                for i, sol in enumerate(solutions):
                    steps.append(f"   ➝ Solution {i+1}: {variable} = {sp.pretty(sol)}")

            return render_template(
                'algebra.html',
                solutions=solutions,
                steps=steps,
                equation=equation,
                variable=variable
            )

        except Exception as e:
            flash(f"Error solving equation: {str(e)}")
            return render_template('algebra.html')

    return render_template('algebra.html')


# MATRICES MODULE
@app.route('/matrices', methods=['GET', 'POST'])
def matrices():
    if request.method == 'POST':
        try:
            # Get matrix size and elements
            size = int(request.form['size'])
            matrix_data = []
            
            for i in range(size):
                row = []
                for j in range(size):
                    value = float(request.form[f'matrix_{i}_{j}'])
                    row.append(value)
                matrix_data.append(row)
            
            # Create numpy matrix
            matrix = np.array(matrix_data)
            
            # Calculate properties
            det = np.linalg.det(matrix)
            rank = np.linalg.matrix_rank(matrix)
            
            try:
                inverse = np.linalg.inv(matrix)
                has_inverse = True
            except np.linalg.LinAlgError:
                inverse = None
                has_inverse = False
            
            try:
                eigenvalues, eigenvectors = np.linalg.eig(matrix)
            except:
                eigenvalues, eigenvectors = None, None
            
            return render_template('matrices.html', 
                                 matrix=matrix.tolist(),
                                 determinant=det,
                                 rank=rank,
                                 inverse=inverse.tolist() if has_inverse else None,
                                 eigenvalues=eigenvalues.tolist() if eigenvalues is not None else None,
                                 size=size)
        except Exception as e:
            flash(f"Error processing matrix: {str(e)}")
            return render_template('matrices.html')
    
    return render_template('matrices.html')

# TRIGONOMETRY MODULE
@app.route('/trigonometry', methods=['GET', 'POST'])
def trigonometry():
    if request.method == 'POST':
        try:
            func_name = request.form['function']
            
            # Create x values
            x = np.linspace(-2*np.pi, 2*np.pi, 1000)
            
            # Calculate y values based on selected function
            func_map = {
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'sec': lambda x: 1/np.cos(x),
                'csc': lambda x: 1/np.sin(x),
                'cot': lambda x: 1/np.tan(x)
            }
            
            y = func_map[func_name](x)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, y, 'b-', linewidth=2, label=f'{func_name}(x)')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            ax.set_xlabel('x (radians)')
            ax.set_ylabel(f'{func_name}(x)')
            ax.set_title(f'Graph of {func_name}(x)')
            ax.legend()
            
            # Handle discontinuities for tan, sec, csc, cot
            if func_name in ['tan', 'sec', 'csc', 'cot']:
                ax.set_ylim(-10, 10)
            
            plot_url = create_plot_base64(fig)
            
            return render_template('trigonometry.html', 
                                 plot_url=plot_url,
                                 function=func_name)
        except Exception as e:
            flash(f"Error plotting function: {str(e)}")
            return render_template('trigonometry.html')
    
    return render_template('trigonometry.html')

# CALCULUS MODULE
@app.route('/calculus', methods=['GET', 'POST'])
def calculus():
    if request.method == 'POST':
        try:
            function_str = request.form['function']
            variable = request.form.get('variable', 'x')
            
            # Parse function
            x = sp.Symbol(variable)
            f = sp.sympify(function_str)
            
            # Calculate derivative and integral
            derivative = sp.diff(f, x)
            integral = sp.integrate(f, x)
            
            # Create plots
            x_vals = np.linspace(-5, 5, 1000)
            f_lambdified = sp.lambdify(x, f, 'numpy')
            df_lambdified = sp.lambdify(x, derivative, 'numpy')
            
            try:
                y_vals = f_lambdified(x_vals)
                dy_vals = df_lambdified(x_vals)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Original function plot
                ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {f}')
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=0, color='k', linewidth=0.5)
                ax1.axvline(x=0, color='k', linewidth=0.5)
                ax1.set_xlabel('x')
                ax1.set_ylabel('f(x)')
                ax1.set_title('Original Function')
                ax1.legend()
                
                # Derivative plot
                ax2.plot(x_vals, dy_vals, 'r-', linewidth=2, label=f"f'(x) = {derivative}")
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='k', linewidth=0.5)
                ax2.axvline(x=0, color='k', linewidth=0.5)
                ax2.set_xlabel('x')
                ax2.set_ylabel("f'(x)")
                ax2.set_title('Derivative')
                ax2.legend()
                
                plot_url = create_plot_base64(fig)
            except:
                plot_url = None
            
            return render_template('calculus.html',
                                 function=function_str,
                                 derivative=str(derivative),
                                 integral=str(integral),
                                 plot_url=plot_url)
        except Exception as e:
            flash(f"Error processing function: {str(e)}")
            return render_template('calculus.html')
    
    return render_template('calculus.html')

# PHYSICS MODULE
@app.route('/physics', methods=['GET', 'POST'])
def physics():
    results = {}
    
    if request.method == 'POST':
        try:
            calc_type = request.form['calc_type']
            
            if calc_type == 'force':
                mass = float(request.form['mass'])
                acceleration = float(request.form['acceleration'])
                force = mass * acceleration
                results = {'type': 'force', 'mass': mass, 'acceleration': acceleration, 'force': force}
                
            elif calc_type == 'wave':
                tension = float(request.form['tension'])
                density = float(request.form['density'])
                wave_speed = np.sqrt(tension / density)
                results = {'type': 'wave', 'tension': tension, 'density': density, 'wave_speed': wave_speed}
                
            elif calc_type == 'ohm':
                voltage = float(request.form['voltage'])
                resistance = float(request.form['resistance'])
                current = voltage / resistance
                power = voltage * current
                results = {'type': 'ohm', 'voltage': voltage, 'resistance': resistance, 'current': current, 'power': power}
                
        except Exception as e:
            flash(f"Error in physics calculation: {str(e)}")
    
    return render_template('physics.html', results=results)

# STATISTICS MODULE
@app.route('/statistics', methods=['GET', 'POST'])
def statistics():
    if request.method == 'POST':
        try:
            if 'csv_file' in request.files:
                file = request.files['csv_file']
                if file.filename != '':
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    # Read CSV
                    df = pd.read_csv(filepath)
                    
                    # Get numeric columns only
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if not numeric_cols:
                        flash("No numeric columns found in CSV file")
                        return render_template('statistics.html')
                    
                    # Calculate statistics for first numeric column
                    col = numeric_cols[0]
                    data = df[col].dropna()
                    
                    stats = {
                        'mean': np.mean(data),
                        'median': np.median(data),
                        'std': np.std(data),
                        'var': np.var(data),
                        'min': np.min(data),
                        'max': np.max(data),
                        'count': len(data)
                    }
                    
                    # Create plots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Histogram
                    ax1.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    ax1.set_xlabel(col)
                    ax1.set_ylabel('Frequency')
                    ax1.set_title(f'Histogram of {col}')
                    ax1.grid(True, alpha=0.3)
                    
                    # Box plot
                    ax2.boxplot(data)
                    ax2.set_ylabel(col)
                    ax2.set_title(f'Box Plot of {col}')
                    ax2.grid(True, alpha=0.3)
                    
                    plot_url = create_plot_base64(fig)
                    
                    # Clean up
                    os.remove(filepath)
                    
                    return render_template('statistics.html', 
                                         stats=stats, 
                                         plot_url=plot_url,
                                         column_name=col)
            else:
                # Manual data entry
                data_str = request.form['manual_data']
                data = [float(x.strip()) for x in data_str.split(',') if x.strip()]
                
                stats = {
                    'mean': np.mean(data),
                    'median': np.median(data),
                    'std': np.std(data),
                    'var': np.var(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'count': len(data)
                }
                
                # Create plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Histogram
                ax1.hist(data, bins=min(20, len(data)//2 + 1), alpha=0.7, color='skyblue', edgecolor='black')
                ax1.set_xlabel('Values')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Histogram')
                ax1.grid(True, alpha=0.3)
                
                # Box plot
                ax2.boxplot(data)
                ax2.set_ylabel('Values')
                ax2.set_title('Box Plot')
                ax2.grid(True, alpha=0.3)
                
                plot_url = create_plot_base64(fig)
                
                return render_template('statistics.html', 
                                     stats=stats, 
                                     plot_url=plot_url,
                                     column_name='Data')
                
        except Exception as e:
            flash(f"Error processing statistics: {str(e)}")
            return render_template('statistics.html')
    
    return render_template('statistics.html')

# ADVANCED MATH MODULE
@app.route('/advanced', methods=['GET', 'POST'])
def advanced():
    plot_html = None
    plot_type = None
    
    if request.method == 'POST':
        try:
            viz_type = request.form['viz_type']
            
            if viz_type == '3d_surface':
                # 3D Surface plot
                function_str = request.form.get('surface_function', 'x**2 + y**2')
                
                x = np.linspace(-5, 5, 50)
                y = np.linspace(-5, 5, 50)
                X, Y = np.meshgrid(x, y)
                
                # Parse and evaluate function
                x_sym, y_sym = sp.symbols('x y')
                f = sp.sympify(function_str)
                f_lambdified = sp.lambdify((x_sym, y_sym), f, 'numpy')
                Z = f_lambdified(X, Y)
                
                # Create 3D plot with Plotly
                fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
                fig.update_layout(
                    title=f'3D Surface: z = {function_str}',
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'
                    ),
                    width=800,
                    height=600
                )
                plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                plot_type = '3D Surface'
                
            elif viz_type == 'fourier':
                # Fourier Series visualization
                n_terms = int(request.form.get('fourier_terms', '5'))
                
                x = np.linspace(-2*np.pi, 2*np.pi, 1000)
                
                # Square wave Fourier series
                y = np.zeros_like(x)
                for n in range(1, n_terms + 1):
                    if n % 2 == 1:  # odd terms only for square wave
                        y += (4 / (n * np.pi)) * np.sin(n * x)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'Fourier Series ({n_terms} terms)'))
                
                # Add actual square wave for comparison
                square_wave = np.sign(np.sin(x))
                fig.add_trace(go.Scatter(x=x, y=square_wave, mode='lines', name='Target Square Wave', opacity=0.5))
                
                fig.update_layout(
                    title=f'Fourier Series Approximation of Square Wave ({n_terms} terms)',
                    xaxis_title='x',
                    yaxis_title='f(x)',
                    width=800,
                    height=500
                )
                plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                plot_type = 'Fourier Series'
                
            elif viz_type == 'zeta':
                # Riemann Zeta function visualization
                s_values = np.linspace(0.1, 10, 100)
                zeta_values = []
                
                for s in s_values:
                    # Approximation using first 1000 terms
                    zeta_approx = sum(1 / (n**s) for n in range(1, 1001))
                    zeta_values.append(zeta_approx)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=s_values, y=zeta_values, mode='lines', name='ζ(s)'))
                fig.update_layout(
                    title='Riemann Zeta Function ζ(s)',
                    xaxis_title='s',
                    yaxis_title='ζ(s)',
                    width=800,
                    height=500
                )
                plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                plot_type = 'Riemann Zeta'
                
        except Exception as e:
            flash(f"Error creating visualization: {str(e)}")
    
    return render_template('advanced.html', plot_html=plot_html, plot_type=plot_type)

if __name__ == '__main__':
    app.run(debug=True)