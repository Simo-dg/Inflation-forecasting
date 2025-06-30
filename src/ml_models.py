from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.utils import train_test_split_time_series
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def run_rf(df, target_col, train_start, train_end, test_start, test_end):
    X_train, y_train, _, y_test = train_test_split_time_series(df, train_start, train_end, test_start, test_end, target_col)
    X_train = X_train.dropna()
    y_train = y_train.dropna()
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    forecasts = []
    history_X = X_train.copy()
    history_y = y_train.copy()

    for t in range(len(y_test)):
        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(history_X, history_y)
        X_step = df.loc[[y_test.index[t]]].drop(columns=[target_col])
        forecasts.append(model.predict(X_step)[0])
        history_X = pd.concat([history_X, X_step])
        history_y = pd.concat([history_y, y_test.iloc[t:t+1]])

    forecasts = np.array(forecasts)
    rmse = np.sqrt(mean_squared_error(y_test, forecasts))

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), y_test.to_numpy(), label="Observed", color="blue")
    plt.plot(y_test.index.to_numpy(), forecasts, label="Forecast", color="red", linestyle="--")
    plt.title(f"Random Forest RMSE: {rmse:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/rf_forecast.png")
    plt.close()

    return forecasts, rmse


def run_xgboost(df, target_col, train_start, train_end, test_start, test_end):
    _, y_train, _, y_test = train_test_split_time_series(df, train_start, train_end, test_start, test_end, target_col)
    X_train = df.loc[train_start:train_end].drop(columns=[target_col]).dropna()
    y_train = y_train.dropna()
    
    # Align train X,y
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)

    forecasts = []
    history_X = X_train.copy()
    history_y = y_train.copy()

    for t in range(len(y_test)):
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(history_X, history_y)

        X_test_step = df.loc[[y_test.index[t]]].drop(columns=[target_col])
        fcast = model.predict(X_test_step)[0]
        forecasts.append(fcast)

        history_X = pd.concat([history_X, X_test_step])
        history_y = pd.concat([history_y, y_test.iloc[t:t+1]])

    forecasts = np.array(forecasts)
    rmse = np.sqrt(mean_squared_error(y_test, forecasts))

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), y_test.values, label="Observed (Test)", color="blue", linewidth=2)
    plt.plot(y_test.index.to_numpy(), forecasts, label="1-step Ahead Forecast Mean", linestyle="--", color="red", linewidth=2)
    plt.title(f"XGBoost (RMSE={rmse:.4f})")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/xgb_forecast.png")
    plt.close()

    return forecasts, rmse


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import networkx as nx
from matplotlib.patches import Circle, FancyBboxPatch
import seaborn as sns
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

class AnimatedNeuralNetwork:
    def __init__(self, layer_sizes, figsize=(16, 10)):
        """
        Create an animated neural network visualization
        
        Args:
            layer_sizes: list of integers representing neurons in each layer
            figsize: figure size tuple
        """
        self.layer_sizes = layer_sizes
        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.neuron_positions = {}
        self.connections = []
        self.neuron_circles = {}
        self.connection_lines = {}
        self.activation_values = {}
        
    def setup_network_layout(self):
        """Calculate positions for neurons and connections"""
        max_neurons = max(self.layer_sizes)
        layer_width = 3.0
        neuron_radius = 0.3
        
        for layer_idx, num_neurons in enumerate(self.layer_sizes):
            x = layer_idx * layer_width
            
            # Center neurons vertically
            if num_neurons == 1:
                y_positions = [0]
            else:
                y_start = -(num_neurons - 1) / 2
                y_positions = [y_start + i for i in range(num_neurons)]
            
            for neuron_idx, y in enumerate(y_positions):
                self.neuron_positions[(layer_idx, neuron_idx)] = (x, y)
        
        # Create connections between layers
        for layer_idx in range(len(self.layer_sizes) - 1):
            for from_neuron in range(self.layer_sizes[layer_idx]):
                for to_neuron in range(self.layer_sizes[layer_idx + 1]):
                    self.connections.append(((layer_idx, from_neuron), (layer_idx + 1, to_neuron)))
    
    def create_static_network(self, weights=None, title="Neural Network Architecture"):
        """Create static network visualization"""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.setup_network_layout()
        
        # Draw connections first (so they appear behind neurons)
        for i, (from_pos, to_pos) in enumerate(self.connections):
            from_x, from_y = self.neuron_positions[from_pos]
            to_x, to_y = self.neuron_positions[to_pos]
            
            # Weight-based line thickness and color
            if weights is not None and i < len(weights):
                weight = weights[i]
                linewidth = abs(weight) * 3 + 0.5
                color = 'red' if weight > 0 else 'blue'
                alpha = min(abs(weight) + 0.2, 1.0)
            else:
                linewidth = 1.0
                color = 'gray'
                alpha = 0.3
            
            line = self.ax.plot([from_x, to_x], [from_y, to_y], 
                              color=color, linewidth=linewidth, alpha=alpha)[0]
            self.connection_lines[i] = line
        
        # Draw neurons
        layer_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        
        for (layer_idx, neuron_idx), (x, y) in self.neuron_positions.items():
            color = layer_colors[layer_idx % len(layer_colors)]
            circle = Circle((x, y), 0.3, color=color, ec='black', linewidth=2, zorder=5)
            self.ax.add_patch(circle)
            self.neuron_circles[(layer_idx, neuron_idx)] = circle
            
            # Add neuron labels
            if layer_idx == 0:
                self.ax.text(x, y, f'X{neuron_idx+1}', ha='center', va='center', 
                           fontweight='bold', fontsize=8)
            elif layer_idx == len(self.layer_sizes) - 1:
                self.ax.text(x, y, 'Y', ha='center', va='center', 
                           fontweight='bold', fontsize=10)
            else:
                self.ax.text(x, y, f'H{neuron_idx+1}', ha='center', va='center', 
                           fontweight='bold', fontsize=8)
        
        # Add layer labels
        for layer_idx, num_neurons in enumerate(self.layer_sizes):
            x = layer_idx * 3.0
            if layer_idx == 0:
                label = f'Input Layer\n({num_neurons} features)'
            elif layer_idx == len(self.layer_sizes) - 1:
                label = f'Output Layer\n({num_neurons} output)'
            else:
                label = f'Hidden Layer {layer_idx}\n({num_neurons} neurons)'
            
            self.ax.text(x, max(self.layer_sizes)/2 + 1, label, ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Styling
        self.ax.set_xlim(-1, (len(self.layer_sizes) - 1) * 3 + 1)
        self.ax.set_ylim(-max(self.layer_sizes)/2 - 1, max(self.layer_sizes)/2 + 2)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        return self.fig, self.ax
    
    def animate_forward_pass(self, input_data, model=None, save_path='results/neural_network_animation.gif'):
        """Create animated forward pass visualization"""
        self.create_static_network(title="Neural Network - Forward Pass Animation")
        
        # Initialize activation values
        for layer_idx in range(len(self.layer_sizes)):
            for neuron_idx in range(self.layer_sizes[layer_idx]):
                self.activation_values[(layer_idx, neuron_idx)] = 0.0
        
        def animate(frame):
            # Clear previous activations
            for circle in self.neuron_circles.values():
                circle.set_facecolor('lightgray')
                circle.set_alpha(0.3)
            
            # Simulate forward pass
            current_layer = frame // 10  # Change layer every 10 frames
            if current_layer >= len(self.layer_sizes):
                current_layer = len(self.layer_sizes) - 1
            
            # Activate neurons up to current layer
            for layer_idx in range(current_layer + 1):
                for neuron_idx in range(self.layer_sizes[layer_idx]):
                    circle = self.neuron_circles[(layer_idx, neuron_idx)]
                    
                    if layer_idx == 0:
                        # Input layer - use actual input values
                        if neuron_idx < len(input_data):
                            activation = abs(input_data[neuron_idx])
                            normalized_activation = min(activation / (np.max(np.abs(input_data)) + 1e-8), 1.0)
                        else:
                            normalized_activation = 0.0
                    else:
                        # Hidden/output layers - simulate activation
                        normalized_activation = np.random.rand() * 0.8 + 0.2
                    
                    # Color intensity based on activation
                    color_intensity = normalized_activation
                    if layer_idx == 0:
                        color = plt.cm.Blues(0.3 + color_intensity * 0.7)
                    elif layer_idx == len(self.layer_sizes) - 1:
                        color = plt.cm.Reds(0.3 + color_intensity * 0.7)
                    else:
                        color = plt.cm.Greens(0.3 + color_intensity * 0.7)
                    
                    circle.set_facecolor(color)
                    circle.set_alpha(0.8)
            
            # Animate connections for current layer transition
            if current_layer > 0:
                for i, (from_pos, to_pos) in enumerate(self.connections):
                    if from_pos[0] == current_layer - 1:
                        line = self.connection_lines[i]
                        # Pulse effect
                        pulse = 0.5 + 0.5 * np.sin(frame * 0.5)
                        line.set_alpha(pulse * 0.8)
                        line.set_linewidth(2 + pulse * 2)
            
            return list(self.neuron_circles.values()) + list(self.connection_lines.values())
        
        # Create animation
        frames = len(self.layer_sizes) * 15
        anim = animation.FuncAnimation(self.fig, animate, frames=frames, 
                                     interval=200, blit=False, repeat=True)
        
        # Save animation
        anim.save(save_path, writer='pillow', fps=5, dpi=100)
        plt.close()
        
        return anim

def create_mlp_architecture_plots(best_params, X_train, y_train, model=None):
    """
    Create comprehensive neural network architecture visualizations
    """
    
    # Extract architecture from best_params
    hidden_layers = best_params.get('hidden_layer_sizes', (50,))
    if isinstance(hidden_layers, int):
        hidden_layers = (hidden_layers,)
    
    input_size = X_train.shape[1] if hasattr(X_train, 'shape') else 10
    output_size = 1
    
    # Full architecture
    layer_sizes = [input_size] + list(hidden_layers) + [output_size]
    
    # 1. Static Network Architecture
    plt.figure(figsize=(16, 10))
    
    # Main architecture plot
    plt.subplot(2, 2, 1)
    nn_viz = AnimatedNeuralNetwork(layer_sizes)
    nn_viz.create_static_network(title="MLP Architecture")
    plt.gca().set_title(f"MLP Architecture: {layer_sizes}", fontsize=14, fontweight='bold')
    
    # 2. Hyperparameter Visualization
    plt.subplot(2, 2, 2)
    
    # Create hyperparameter summary
    param_names = []
    param_values = []
    
    for param, value in best_params.items():
        param_names.append(param.replace('_', '\n'))
        if isinstance(value, tuple):
            param_values.append(len(value))  # For hidden layer sizes, show number of layers
        elif isinstance(value, (int, float)):
            param_values.append(value)
        else:
            param_values.append(1)  # Default for other types
    
    bars = plt.bar(range(len(param_names)), param_values, 
                   color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'][:len(param_names)])
    plt.xticks(range(len(param_names)), param_names, rotation=45, ha='right')
    plt.ylabel('Parameter Value')
    plt.title('Best Hyperparameters', fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, param_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_values)*0.01,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # 3. Network Complexity Analysis
    plt.subplot(2, 2, 3)
    
    # Calculate network statistics
    total_params = 0
    layer_params = []
    
    for i in range(len(layer_sizes) - 1):
        params = layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]  # weights + biases
        layer_params.append(params)
        total_params += params
    
    layer_names = [f'Layer {i+1}\nto {i+2}' for i in range(len(layer_params))]
    
    plt.pie(layer_params, labels=layer_names, autopct='%1.1f%%', startangle=90)
    plt.title(f'Parameter Distribution\nTotal Parameters: {total_params:,}', fontweight='bold')
    
    # 4. Activation Function Visualization
    plt.subplot(2, 2, 4)
    
    x = np.linspace(-5, 5, 100)
    
    # Common activation functions
    relu = np.maximum(0, x)
    tanh = np.tanh(x)
    sigmoid = 1 / (1 + np.exp(-x))
    
    plt.plot(x, relu, label='ReLU', linewidth=2)
    plt.plot(x, tanh, label='Tanh', linewidth=2)
    plt.plot(x, sigmoid, label='Sigmoid', linewidth=2)
    
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Activation Functions\n(ReLU typically used in hidden layers)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/mlp_architecture_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Create detailed architecture diagram
    create_detailed_architecture_diagram(layer_sizes, best_params)
    
    # 6. Create animated forward pass
    if input_size <= 10:  # Only animate for smaller networks
        sample_input = np.random.randn(min(input_size, 8))
        nn_anim = AnimatedNeuralNetwork(layer_sizes[:4] if len(layer_sizes) > 4 else layer_sizes)
        try:
            anim = nn_anim.animate_forward_pass(sample_input)
            print("✅ Neural network animation saved as 'results/neural_network_animation.gif'")
        except Exception as e:
            print(f"⚠️ Animation creation failed: {e}")
    
    return layer_sizes, total_params

def create_detailed_architecture_diagram(layer_sizes, best_params):
    """Create a detailed technical architecture diagram"""
    
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Draw detailed network with mathematical annotations
    y_positions = []
    layer_width = 4.0
    max_display_neurons = 8  # Limit displayed neurons for clarity
    
    # Calculate positions
    for layer_idx, num_neurons in enumerate(layer_sizes):
        x = layer_idx * layer_width
        display_neurons = min(num_neurons, max_display_neurons)
        
        if display_neurons == 1:
            y_pos = [0]
        else:
            y_start = -(display_neurons - 1) / 2
            y_pos = [y_start + i for i in range(display_neurons)]
        
        y_positions.append(y_pos)
        
        # Draw neurons
        for i, y in enumerate(y_pos):
            if layer_idx == 0:
                color = 'lightblue'
                label = f'x₁' if i == 0 else f'x_{i+1}' if i < 3 else '⋮' if i == 3 else f'x_{num_neurons}'
            elif layer_idx == len(layer_sizes) - 1:
                color = 'lightcoral'
                label = 'ŷ'
            else:
                color = 'lightgreen'
                label = f'h₁⁽{layer_idx}⁾' if i == 0 else f'h_{i+1}⁽{layer_idx}⁾' if i < 3 else '⋮' if i == 3 else f'h_{display_neurons}⁽{layer_idx}⁾'
            
            circle = Circle((x, y), 0.25, color=color, ec='black', linewidth=1.5)
            ax.add_patch(circle)
            
            if i != 3:  # Don't label the ellipsis
                ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Add ellipsis if we're not showing all neurons
        if num_neurons > max_display_neurons:
            ax.text(x, y_pos[3], '⋮', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Draw connections with mathematical notation
    for layer_idx in range(len(layer_sizes) - 1):
        from_positions = y_positions[layer_idx]
        to_positions = y_positions[layer_idx + 1]
        
        from_x = layer_idx * layer_width
        to_x = (layer_idx + 1) * layer_width
        
        # Draw sample connections
        for i, from_y in enumerate(from_positions[:3]):  # Only first 3 for clarity
            for j, to_y in enumerate(to_positions[:3]):
                ax.plot([from_x, to_x], [from_y, to_y], 'gray', alpha=0.4, linewidth=0.8)
                
                # Add weight annotation on first few connections
                if i == 0 and j == 0:
                    mid_x = (from_x + to_x) / 2
                    mid_y = (from_y + to_y) / 2
                    ax.text(mid_x, mid_y + 0.1, f'w₁₁⁽{layer_idx+1}⁾', 
                           ha='center', va='center', fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Add mathematical formulas
    formula_y = max([max(pos) for pos in y_positions]) + 1.5
    
    # Forward pass equation
    ax.text(layer_width * (len(layer_sizes) - 1) / 2, formula_y + 1, 
           'Forward Pass: h⁽ˡ⁺¹⁾ = f(W⁽ˡ⁺¹⁾h⁽ˡ⁾ + b⁽ˡ⁺¹⁾)', 
           ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    # Add layer information
    for layer_idx, num_neurons in enumerate(layer_sizes):
        x = layer_idx * layer_width
        y = min([min(pos) for pos in y_positions]) - 1
        
        if layer_idx == 0:
            layer_info = f'Input Layer\n{num_neurons} features'
        elif layer_idx == len(layer_sizes) - 1:
            layer_info = f'Output Layer\n{num_neurons} output'
        else:
            layer_info = f'Hidden Layer {layer_idx}\n{num_neurons} neurons\nActivation: ReLU'
        
        ax.text(x, y, layer_info, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Add hyperparameter information
    param_text = "Best Hyperparameters:\n"
    for param, value in best_params.items():
        param_text += f"{param}: {value}\n"
    
    ax.text(layer_width * (len(layer_sizes) - 1) + 1, formula_y, param_text,
           ha='left', va='top', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # Styling
    ax.set_xlim(-1, layer_width * (len(layer_sizes) - 1) + 3)
    ax.set_ylim(min([min(pos) for pos in y_positions]) - 2, formula_y + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Detailed MLP Architecture with Mathematical Notation', 
                fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/detailed_mlp_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_training_process_animation(training_history=None):
    """Create animation showing the training process"""
    
    # Simulated training history if not provided
    if training_history is None:
        epochs = 100
        training_history = {
            'loss': np.exp(-np.linspace(0, 3, epochs)) + 0.1 * np.random.randn(epochs) * 0.1,
            'val_loss': np.exp(-np.linspace(0, 2.8, epochs)) + 0.15 * np.random.randn(epochs) * 0.1
        }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Training loss animation
    def animate_training(frame):
        if frame == 0:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
        
        # Plot 1: Training Loss
        epochs = range(1, frame + 2)
        ax1.plot(epochs, training_history['loss'][:frame + 1], 'b-', linewidth=2, label='Training Loss')
        ax1.plot(epochs, training_history['val_loss'][:frame + 1], 'r--', linewidth=2, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training Progress (Epoch {frame + 1})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, len(training_history['loss']))
        ax1.set_ylim(0, max(max(training_history['loss']), max(training_history['val_loss'])) * 1.1)
        
        # Plot 2: Weight Evolution (simulated)
        weights = np.random.randn(frame + 1) * np.exp(-frame * 0.05)
        ax2.bar(range(len(weights[-10:])), weights[-10:], alpha=0.7)
        ax2.set_title('Recent Weight Updates')
        ax2.set_ylabel('Weight Magnitude')
        ax2.set_xlabel('Weight Index')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Gradient Norm (simulated)
        grad_norms = np.exp(-np.linspace(0, 2, frame + 1)) + 0.1 * np.random.randn(frame + 1) * 0.05
        ax3.plot(range(1, frame + 2), grad_norms, 'g-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_title('Gradient Magnitude')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(1, len(training_history['loss']))
        
        # Plot 4: Learning Rate Schedule (if applicable)
        lr_schedule = 0.01 * np.exp(-np.linspace(0, 1, frame + 1))
        ax4.plot(range(1, frame + 2), lr_schedule, 'm-', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(1, len(training_history['loss']))
    
    # Create animation
    frames = min(len(training_history['loss']), 50)  # Limit frames for reasonable file size
    anim = animation.FuncAnimation(fig, animate_training, frames=frames, 
                                 interval=200, repeat=True)
    
    try:
        anim.save('results/training_process_animation.gif', writer='pillow', fps=5, dpi=100)
        print("✅ Training process animation saved as 'results/training_process_animation.gif'")
    except Exception as e:
        print(f"⚠️ Training animation creation failed: {e}")
    
    plt.close()
    return anim

# Usage example - add this to your main MLP function:
def create_all_network_visualizations(best_params, X_train, y_train, model=None):
    """
    Create all neural network visualizations
    """
    print("🧠 Creating neural network visualizations...")
    
    # 1. Architecture analysis
    layer_sizes, total_params = create_mlp_architecture_plots(best_params, X_train, y_train, model)
    
    # 2. Training process animation
    create_training_process_animation()
    
    # 3. Interactive network diagram (static version)
    if len(layer_sizes) <= 5:  # Only for reasonably sized networks
        nn_viz = AnimatedNeuralNetwork(layer_sizes)
        fig, ax = nn_viz.create_static_network(title="MLP Architecture Overview")
        plt.savefig('results/mlp_network_diagram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Network visualization complete!")
    print(f"   📊 Architecture: {layer_sizes}")
    print(f"   🔧 Total Parameters: {total_params:,}")
    print(f"   💾 Files saved in 'results/' directory")
    
    return layer_sizes, total_params


def run_mlp(df, target_col, train_start, train_end, test_start, test_end):
    X_train, y_train, _, y_test = train_test_split_time_series(df, train_start, train_end, test_start, test_end, target_col)
    X_train = X_train.dropna()
    y_train = y_train.dropna()
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(MLPRegressor(max_iter=1000, random_state=42), param_grid, cv=tscv,
                        scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_params = grid.best_params_

    forecasts = []
    history_X = X_train.copy()
    history_y = y_train.copy()

    for t in range(len(y_test)):
        history_X_scaled = scaler.fit_transform(history_X)
        model = MLPRegressor(**best_params, max_iter=1000, random_state=42)
        model.fit(history_X_scaled, history_y)

        X_step = df.loc[[y_test.index[t]]].drop(columns=[target_col])
        X_step_scaled = scaler.transform(X_step)

        forecasts.append(model.predict(X_step_scaled)[0])
        history_X = pd.concat([history_X, X_step])
        history_y = pd.concat([history_y, y_test.iloc[t:t+1]])

    forecasts = np.array(forecasts)
    rmse = np.sqrt(mean_squared_error(y_test, forecasts))

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), y_test.to_numpy(), label="Observed", color="blue")
    plt.plot(y_test.index.to_numpy(), forecasts, label="Forecast", color="red", linestyle="--")
    plt.title(f"MLP RMSE: {rmse:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/mlp_forecast.png")
    plt.close()

    layer_sizes, total_params = create_all_network_visualizations(
    best_params, X_train, y_train, grid.best_estimator_
    )
    return forecasts, rmse
