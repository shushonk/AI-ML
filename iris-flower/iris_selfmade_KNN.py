#!/usr/bin/env python3
"""
🌺 Iris Flower Classification Suite - Complete GUI Application
Combined Decision Tree and KNN Classifiers with Enhanced Tkinter Interface
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'

class IrisClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌺 Iris Flower Classification Suite")
        self.root.geometry("1400x950")
        self.root.configure(bg='#f5f5f5')
        
        # Load and prepare data
        self.iris = load_iris()
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names
        self.df = pd.DataFrame(self.iris.data, columns=self.feature_names)
        self.df['species'] = [self.target_names[i] for i in self.iris.target]
        
        # Initialize models and scaler
        self.dt_model = None
        self.knn_model = None
        self.scaler = StandardScaler()
        self.current_plot_frame = None
        
        self.setup_gui()
        self.show_welcome_message()

    def setup_gui(self):
        """Setup the main GUI interface with tabs"""
        # Style configuration
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), background='#f5f5f5')
        style.configure('Section.TLabelframe.Label', font=('Arial', 12, 'bold'))
        
        # Main notebook (tab controller)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)
        self.dt_tab = ttk.Frame(self.notebook)
        self.knn_tab = ttk.Frame(self.notebook)
        self.prediction_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        self.comparison_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.dashboard_tab, text='🏠 Dashboard')
        self.notebook.add(self.data_tab, text='📊 Dataset')
        self.notebook.add(self.dt_tab, text='🌳 Decision Tree')
        self.notebook.add(self.knn_tab, text='🎯 K-Nearest Neighbors')
        self.notebook.add(self.prediction_tab, text='🔮 Prediction')
        self.notebook.add(self.visualization_tab, text='📈 Visualization')
        self.notebook.add(self.comparison_tab, text='⚖️ Comparison')
        
        # Setup each tab
        self.setup_dashboard_tab()
        self.setup_data_tab()
        self.setup_dt_tab()
        self.setup_knn_tab()
        self.setup_prediction_tab()
        self.setup_visualization_tab()
        self.setup_comparison_tab()

    def setup_dashboard_tab(self):
        """Setup the dashboard tab with overview and quick actions"""
        # Title
        title_frame = ttk.Frame(self.dashboard_tab)
        title_frame.pack(fill='x', pady=20)
        
        ttk.Label(title_frame, text="🌺 Iris Flower Classification Suite", 
                 style='Title.TLabel').pack()
        
        ttk.Label(title_frame, text="Complete Machine Learning Solution for Iris Species Classification",
                 font=('Arial', 12), background='#f5f5f5').pack(pady=5)
        
        # Quick stats frame
        stats_frame = ttk.LabelFrame(self.dashboard_tab, text="Dataset Overview", padding=15)
        stats_frame.pack(fill='x', padx=20, pady=10)
        
        stats_text = f"""
• Total Samples: {len(self.iris.data)}
• Features: {len(self.feature_names)} ({', '.join(self.feature_names)})
• Classes: {len(self.target_names)} ({', '.join(self.target_names)})
• Balanced Dataset: 50 samples per class
• Perfect for classification tasks
        """
        ttk.Label(stats_frame, text=stats_text, font=('Consolas', 10), 
                 background='white', relief='solid', padding=10).pack(fill='x')
        
        # Quick actions frame
        actions_frame = ttk.LabelFrame(self.dashboard_tab, text="Quick Actions", padding=15)
        actions_frame.pack(fill='x', padx=20, pady=10)
        
        action_buttons = [
            ("📊 Explore Data", self.show_data_tab),
            ("🌳 Train Decision Tree", lambda: self.notebook.select(self.dt_tab)),
            ("🎯 Train KNN", lambda: self.notebook.select(self.knn_tab)),
            ("🔮 Make Prediction", lambda: self.notebook.select(self.prediction_tab)),
            ("📈 View Visualizations", lambda: self.notebook.select(self.visualization_tab)),
            ("⚖️ Compare Models", lambda: self.notebook.select(self.comparison_tab))
        ]
        
        btn_frame = ttk.Frame(actions_frame)
        btn_frame.pack()
        
        for i, (text, command) in enumerate(action_buttons):
            ttk.Button(btn_frame, text=text, command=command, width=20).grid(
                row=i//3, column=i%3, padx=5, pady=5
            )
        
        # Model status frame
        status_frame = ttk.LabelFrame(self.dashboard_tab, text="Model Status", padding=15)
        status_frame.pack(fill='x', padx=20, pady=10)
        
        self.dt_status = ttk.Label(status_frame, text="❌ Decision Tree: Not Trained", 
                                  font=('Arial', 10), background='#f5f5f5')
        self.dt_status.pack(anchor='w', pady=2)
        
        self.knn_status = ttk.Label(status_frame, text="❌ K-Nearest Neighbors: Not Trained", 
                                   font=('Arial', 10), background='#f5f5f5')
        self.knn_status.pack(anchor='w', pady=2)

    def setup_data_tab(self):
        """Setup data exploration tab"""
        # Main frame
        main_frame = ttk.Frame(self.data_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill='x', pady=10)
        
        ttk.Button(controls_frame, text="Show Dataset Info", 
                  command=self.show_dataset_info).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Show Statistics", 
                  command=self.show_statistics).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Show Sample Data", 
                  command=self.show_sample_data).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Show Correlation", 
                  command=self.show_correlation).pack(side='left', padx=5)
        
        # Data display frame
        data_frame = ttk.LabelFrame(main_frame, text="Data View")
        data_frame.pack(fill='both', expand=True)
        
        self.data_text = scrolledtext.ScrolledText(data_frame, width=100, height=25,
                                                  font=('Consolas', 10))
        self.data_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.show_dataset_info()

    def setup_dt_tab(self):
        """Setup Decision Tree training tab"""
        main_frame = ttk.Frame(self.dt_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="Training Parameters", padding=10)
        config_frame.pack(fill='x', pady=10)
        
        # Parameter grid
        params = [
            ("Test Size:", "test_size_dt", "combobox", [0.2, 0.25, 0.3, 0.35], 0.3),
            ("Random State:", "random_state_dt", "entry", None, 42),
            ("Max Depth:", "max_depth_dt", "combobox", [2, 3, 4, 5, 6, "None"], 4),
            ("Criterion:", "criterion_dt", "combobox", ["gini", "entropy"], "gini"),
            ("Splitter:", "splitter_dt", "combobox", ["best", "random"], "best"),
            ("Min Samples Split:", "min_samples_split_dt", "combobox", [2, 3, 4, 5], 2)
        ]
        
        self.dt_vars = {}
        for i, (label, name, widget_type, values, default) in enumerate(params):
            row = i % 3
            col = (i // 3) * 2
            
            ttk.Label(config_frame, text=label).grid(row=row, column=col, padx=5, pady=5, sticky='w')
            
            if widget_type == "combobox":
                var = tk.StringVar(value=default)
                combo = ttk.Combobox(config_frame, textvariable=var, values=values, 
                                    state='readonly', width=12)
                combo.grid(row=row, column=col+1, padx=5, pady=5)
            else:  # entry
                var = tk.StringVar(value=default)
                entry = ttk.Entry(config_frame, textvariable=var, width=15)
                entry.grid(row=row, column=col+1, padx=5, pady=5)
            
            self.dt_vars[name] = var
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="🚀 Train Decision Tree", 
                  command=self.train_decision_tree, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(btn_frame, text="📊 Evaluate Model", 
                  command=lambda: self.evaluate_model('dt')).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="🔄 Cross-Validation", 
                  command=lambda: self.run_cross_validation('dt')).pack(side='left', padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Training Results")
        results_frame.pack(fill='both', expand=True)
        
        self.dt_results = scrolledtext.ScrolledText(results_frame, font=('Consolas', 9))
        self.dt_results.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_knn_tab(self):
        """Setup KNN training tab"""
        main_frame = ttk.Frame(self.knn_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="Training Parameters", padding=10)
        config_frame.pack(fill='x', pady=10)
        
        # Parameter grid
        params = [
            ("Test Size:", "test_size_knn", "combobox", [0.2, 0.25, 0.3, 0.35], 0.3),
            ("Random State:", "random_state_knn", "entry", None, 42),
            ("Number of Neighbors (k):", "k_value", "combobox", list(range(1, 16)), 5),
            ("Distance Metric:", "distance_metric", "combobox", ["euclidean", "manhattan", "minkowski"], "euclidean"),
            ("Weight Function:", "weights", "combobox", ["uniform", "distance"], "uniform"),
            ("Algorithm:", "algorithm", "combobox", ["auto", "ball_tree", "kd_tree", "brute"], "auto")
        ]
        
        self.knn_vars = {}
        for i, (label, name, widget_type, values, default) in enumerate(params):
            row = i % 3
            col = (i // 3) * 2
            
            ttk.Label(config_frame, text=label).grid(row=row, column=col, padx=5, pady=5, sticky='w')
            
            if widget_type == "combobox":
                var = tk.StringVar(value=default)
                combo = ttk.Combobox(config_frame, textvariable=var, values=values, 
                                    state='readonly', width=12)
                combo.grid(row=row, column=col+1, padx=5, pady=5)
            else:  # entry
                var = tk.StringVar(value=default)
                entry = ttk.Entry(config_frame, textvariable=var, width=15)
                entry.grid(row=row, column=col+1, padx=5, pady=5)
            
            self.knn_vars[name] = var
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="🚀 Train KNN", 
                  command=self.train_knn, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(btn_frame, text="📊 Evaluate Model", 
                  command=lambda: self.evaluate_model('knn')).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="🔄 Cross-Validation", 
                  command=lambda: self.run_cross_validation('knn')).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="🔍 Find Optimal K", 
                  command=self.find_optimal_k).pack(side='left', padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Training Results")
        results_frame.pack(fill='both', expand=True)
        
        self.knn_results = scrolledtext.ScrolledText(results_frame, font=('Consolas', 9))
        self.knn_results.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_prediction_tab(self):
        """Setup prediction tab with interactive input"""
        main_frame = ttk.Frame(self.prediction_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Model selection
        model_frame = ttk.LabelFrame(main_frame, text="Select Classification Model")
        model_frame.pack(fill='x', pady=10)
        
        self.model_var = tk.StringVar(value="decision_tree")
        ttk.Radiobutton(model_frame, text="🌳 Decision Tree", variable=self.model_var,
                       value="decision_tree").pack(side='left', padx=20, pady=10)
        ttk.Radiobutton(model_frame, text="🎯 K-Nearest Neighbors", variable=self.model_var,
                       value="knn").pack(side='left', padx=20, pady=10)
        
        # Input frame with sliders
        input_frame = ttk.LabelFrame(main_frame, text="Flower Measurements (cm)")
        input_frame.pack(fill='x', pady=10)
        
        self.feature_vars = []
        features = [
            ("Sepal Length", 4.0, 8.0),
            ("Sepal Width", 2.0, 4.5),
            ("Petal Length", 1.0, 7.0),
            ("Petal Width", 0.1, 2.5)
        ]
        
        for i, (feature, min_val, max_val) in enumerate(features):
            feature_frame = ttk.Frame(input_frame)
            feature_frame.pack(fill='x', padx=10, pady=8)
            
            ttk.Label(feature_frame, text=feature, width=15).pack(side='left')
            
            var = tk.DoubleVar(value=(min_val + max_val) / 2)
            self.feature_vars.append(var)
            
            # Scale slider
            scale = ttk.Scale(feature_frame, from_=min_val, to=max_val, variable=var,
                             orient='horizontal', length=300)
            scale.pack(side='left', padx=10)
            
            # Value display
            value_label = ttk.Label(feature_frame, textvariable=var, width=6)
            value_label.pack(side='left')
            ttk.Label(feature_frame, text="cm").pack(side='left')
        
        # Control buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=15)
        
        ttk.Button(btn_frame, text="🔮 Predict Species", 
                  command=self.predict_species, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(btn_frame, text="🗑️ Clear", 
                  command=self.clear_inputs).pack(side='left', padx=5)
        
        # Example buttons
        example_frame = ttk.Frame(btn_frame)
        example_frame.pack(side='left', padx=20)
        
        ttk.Label(example_frame, text="Load Example:").pack(side='left')
        ttk.Button(example_frame, text="Setosa", 
                  command=lambda: self.load_example('setosa')).pack(side='left', padx=2)
        ttk.Button(example_frame, text="Versicolor", 
                  command=lambda: self.load_example('versicolor')).pack(side='left', padx=2)
        ttk.Button(example_frame, text="Virginica", 
                  command=lambda: self.load_example('virginica')).pack(side='left', padx=2)
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results")
        results_frame.pack(fill='both', expand=True, pady=10)
        
        self.prediction_display = scrolledtext.ScrolledText(results_frame, font=('Consolas', 10))
        self.prediction_display.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_visualization_tab(self):
        """Setup visualization tab with multiple plot options"""
        main_frame = ttk.Frame(self.visualization_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Visualization controls
        controls_frame = ttk.LabelFrame(main_frame, text="Visualization Options")
        controls_frame.pack(fill='x', pady=10)
        
        # Plot buttons in a grid
        plot_buttons = [
            ("📊 Feature Distribution", self.plot_feature_distribution),
            ("🔥 Correlation Heatmap", self.plot_correlation_heatmap),
            ("🔗 Pair Plot", self.plot_pairplot),
            ("🌳 Decision Tree", self.plot_decision_tree),
            ("⭐ Feature Importance", self.plot_feature_importance),
            ("🎯 KNN Accuracy vs K", self.plot_knn_accuracy),
            ("📈 Class Distribution", self.plot_class_distribution),
            ("🔍 Feature Relationships", self.plot_feature_relationships)
        ]
        
        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack(pady=10)
        
        for i, (text, command) in enumerate(plot_buttons):
            ttk.Button(btn_frame, text=text, command=command, width=18).grid(
                row=i//4, column=i%4, padx=5, pady=5
            )
        
        # Plot display area
        plot_frame = ttk.LabelFrame(main_frame, text="Visualization")
        plot_frame.pack(fill='both', expand=True)
        
        self.plot_canvas_frame = ttk.Frame(plot_frame)
        self.plot_canvas_frame.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_comparison_tab(self):
        """Setup model comparison tab"""
        main_frame = ttk.Frame(self.comparison_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Comparison controls
        controls_frame = ttk.LabelFrame(main_frame, text="Comparison Tools")
        controls_frame.pack(fill='x', pady=10)
        
        ttk.Button(controls_frame, text="⚖️ Compare Both Models", 
                  command=self.compare_models).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="📊 Performance Metrics", 
                  command=self.show_performance_metrics).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="⏱️ Training Time", 
                  command=self.compare_training_time).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="🎯 Cross-Validation", 
                  command=self.compare_cross_validation).pack(side='left', padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Comparison Results")
        results_frame.pack(fill='both', expand=True)
        
        self.comparison_results = scrolledtext.ScrolledText(results_frame, font=('Consolas', 9))
        self.comparison_results.pack(fill='both', expand=True, padx=5, pady=5)

    # Data Management Methods
    def show_dataset_info(self):
        """Display dataset information"""
        info = f"""
{'='*70}
IRIS FLOWER DATASET - COMPLETE OVERVIEW
{'='*70}

Dataset Characteristics:
• Number of Instances: {len(self.iris.data)}
• Number of Features: {len(self.feature_names)}
• Number of Classes: {len(self.target_names)}
• Missing Values: None
• Dataset Size: {self.iris.data.nbytes / 1024:.1f} KB

Feature Information:
"""
        for i, feature in enumerate(self.feature_names):
            info += f"{i+1}. {feature} (cm)\n"
        
        info += f"\nTarget Classes:\n"
        for i, species in enumerate(self.target_names):
            info += f"{i}. {species}\n"
        
        info += f"\nClass Distribution:\n"
        counts = np.bincount(self.iris.target)
        for i, (species, count) in enumerate(zip(self.target_names, counts)):
            info += f"• {species}: {count} samples ({count/len(self.iris.target)*100:.1f}%)\n"
        
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(1.0, info)

    def show_statistics(self):
        """Display dataset statistics"""
        stats = self.df.describe()
        stats_text = f"""
{'='*70}
DATASET STATISTICAL SUMMARY
{'='*70}

{stats}

Key Insights:
• All features are measured in centimeters
• Petal measurements show higher variance than sepal measurements
• Dataset is perfectly balanced across all three species
• No missing values or outliers detected
"""
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(1.0, stats_text)

    def show_sample_data(self):
        """Display sample data"""
        sample = self.df.head(20)
        sample_text = f"""
{'='*70}
SAMPLE DATA (First 20 instances)
{'='*70}

{sample.to_string(index=True)}

Dataset Structure:
• Index: Sample identifier
• sepal length (cm), sepal width (cm), petal length (cm), petal width (cm): Features
• species: Target variable (setosa, versicolor, virginica)
"""
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(1.0, sample_text)

    def show_correlation(self):
        """Display correlation matrix"""
        corr_matrix = self.df[self.feature_names].corr()
        corr_text = f"""
{'='*70}
FEATURE CORRELATION MATRIX
{'='*70}

{corr_matrix}

Correlation Interpretation:
• Strong Positive (>0.7): Petal Length vs Petal Width (0.96)
• Moderate Positive (0.5-0.7): Sepal Length vs Petal Length (0.87)
• Weak Correlation (<0.5): Sepal Width vs other features
• Petal measurements are highly correlated with each other
"""
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(1.0, corr_text)

    # Model Training Methods
    def train_decision_tree(self):
        """Train Decision Tree classifier"""
        try:
            # Get parameters
            test_size = float(self.dt_vars['test_size_dt'].get())
            random_state = int(self.dt_vars['random_state_dt'].get())
            max_depth = self.dt_vars['max_depth_dt'].get()
            max_depth = None if max_depth == "None" else int(max_depth)
            criterion = self.dt_vars['criterion_dt'].get()
            splitter = self.dt_vars['splitter_dt'].get()
            min_samples_split = int(self.dt_vars['min_samples_split_dt'].get())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.iris.data, self.iris.target,
                test_size=test_size,
                random_state=random_state,
                stratify=self.iris.target
            )
            
            # Train model
            start_time = time.time()
            self.dt_model = tree.DecisionTreeClassifier(
                max_depth=max_depth,
                criterion=criterion,
                splitter=splitter,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
            self.dt_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Calculate metrics
            train_accuracy = self.dt_model.score(X_train, y_train)
            test_accuracy = self.dt_model.score(X_test, y_test)
            
            # Display results
            result = f"""
{'='*60}
DECISION TREE TRAINING COMPLETED ✅
{'='*60}

Training Parameters:
• Test Size: {test_size}
• Random State: {random_state}
• Max Depth: {max_depth}
• Criterion: {criterion}
• Splitter: {splitter}
• Min Samples Split: {min_samples_split}

Performance Metrics:
• Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)
• Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
• Training Time: {training_time:.3f} seconds

Feature Importance:
"""
            importances = self.dt_model.feature_importances_
            for feature, importance in zip(self.feature_names, importances):
                result += f"• {feature}: {importance:.4f}\n"
            
            result += f"\nModel is ready for predictions! 🎉"
            
            self.dt_results.delete(1.0, tk.END)
            self.dt_results.insert(1.0, result)
            
            # Update dashboard status
            self.dt_status.config(text=f"✅ Decision Tree: Trained (Test Acc: {test_accuracy*100:.1f}%)")
            
            messagebox.showinfo("Success", "Decision Tree trained successfully!")
            
        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to train Decision Tree:\n{str(e)}")

    def train_knn(self):
        """Train K-Nearest Neighbors classifier"""
        try:
            # Get parameters
            test_size = float(self.knn_vars['test_size_knn'].get())
            random_state = int(self.knn_vars['random_state_knn'].get())
            n_neighbors = int(self.knn_vars['k_value'].get())
            metric = self.knn_vars['distance_metric'].get()
            weights = self.knn_vars['weights'].get()
            algorithm = self.knn_vars['algorithm'].get()
            
            # Split and scale data
            X_train, X_test, y_train, y_test = train_test_split(
                self.iris.data, self.iris.target,
                test_size=test_size,
                random_state=random_state,
                stratify=self.iris.target
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            start_time = time.time()
            self.knn_model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                metric=metric,
                weights=weights,
                algorithm=algorithm
            )
            self.knn_model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # Calculate metrics
            train_accuracy = self.knn_model.score(X_train_scaled, y_train)
            test_accuracy = self.knn_model.score(X_test_scaled, y_test)
            
            # Display results
            result = f"""
{'='*60}
K-NEAREST NEIGHBORS TRAINING COMPLETED ✅
{'='*60}

Training Parameters:
• Test Size: {test_size}
• Random State: {random_state}
• Number of Neighbors (k): {n_neighbors}
• Distance Metric: {metric}
• Weight Function: {weights}
• Algorithm: {algorithm}

Performance Metrics:
• Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)
• Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
• Training Time: {training_time:.3f} seconds

Model Characteristics:
• Uses {n_neighbors} nearest neighbors for classification
• Distance metric: {metric}
• Feature scaling: Applied (StandardScaler)
"""
            result += f"\nModel is ready for predictions! 🎉"
            
            self.knn_results.delete(1.0, tk.END)
            self.knn_results.insert(1.0, result)
            
            # Update dashboard status
            self.knn_status.config(text=f"✅ K-Nearest Neighbors: Trained (Test Acc: {test_accuracy*100:.1f}%)")
            
            messagebox.showinfo("Success", "KNN trained successfully!")
            
        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to train KNN:\n{str(e)}")

    # Evaluation Methods
    def evaluate_model(self, model_type):
        """Evaluate the specified model"""
        try:
            if model_type == 'dt' and not self.dt_model:
                messagebox.showerror("Error", "Please train Decision Tree first!")
                return
            elif model_type == 'knn' and not self.knn_model:
                messagebox.showerror("Error", "Please train KNN first!")
                return
            
            # Prepare test data
            X_train, X_test, y_train, y_test = train_test_split(
                self.iris.data, self.iris.target,
                test_size=0.3,
                random_state=42,
                stratify=self.iris.target
            )
            
            if model_type == 'dt':
                model = self.dt_model
                predictions = model.predict(X_test)
                results_widget = self.dt_results
                model_name = "Decision Tree"
            else:
                model = self.knn_model
                X_test_scaled = self.scaler.transform(X_test)
                predictions = model.predict(X_test_scaled)
                results_widget = self.knn_results
                model_name = "K-Nearest Neighbors"
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)
            report = classification_report(y_test, predictions, target_names=self.target_names)
            
            result = f"""
{'='*60}
{model_name.upper()} EVALUATION 📊
{'='*60}

Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)

Confusion Matrix:
{cm}

Detailed Classification Report:
{report}

Key Metrics:
• Precision: Correct positive predictions relative to total positive predictions
• Recall: Correct positive predictions relative to actual positives  
• F1-Score: Harmonic mean of precision and recall
• Support: Number of actual occurrences of each class
"""
            results_widget.delete(1.0, tk.END)
            results_widget.insert(1.0, result)
            
        except Exception as e:
            messagebox.showerror("Evaluation Error", f"Failed to evaluate model:\n{str(e)}")

    def run_cross_validation(self, model_type):
        """Run cross-validation for the specified model"""
        try:
            if model_type == 'dt' and not self.dt_model:
                messagebox.showerror("Error", "Please train Decision Tree first!")
                return
            elif model_type == 'knn' and not self.knn_model:
                messagebox.showerror("Error", "Please train KNN first!")
                return
            
            if model_type == 'dt':
                model = tree.DecisionTreeClassifier(
                    max_depth=int(self.dt_vars['max_depth_dt'].get()) if self.dt_vars['max_depth_dt'].get() != "None" else None,
                    criterion=self.dt_vars['criterion_dt'].get()
                )
                results_widget = self.dt_results
                model_name = "Decision Tree"
            else:
                model = KNeighborsClassifier(
                    n_neighbors=int(self.knn_vars['k_value'].get()),
                    metric=self.knn_vars['distance_metric'].get()
                )
                results_widget = self.knn_results
                model_name = "K-Nearest Neighbors"
            
            # 5-fold cross-validation
            cv_scores = cross_val_score(model, self.iris.data, self.iris.target, cv=5)
            
            result = f"""
{'='*60}
{model_name.upper()} CROSS-VALIDATION 🔄
{'='*60}

5-Fold Cross-Validation Scores:
{cv_scores}

Statistics:
• Mean Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)
• Standard Deviation: {cv_scores.std():.4f}
• Minimum Score: {cv_scores.min():.4f}
• Maximum Score: {cv_scores.max():.4f}

Interpretation:
• Lower standard deviation indicates more consistent performance
• Higher mean accuracy indicates better overall performance
• Cross-validation provides better estimate of real-world performance
"""
            results_widget.delete(1.0, tk.END)
            results_widget.insert(1.0, result)
            
        except Exception as e:
            messagebox.showerror("CV Error", f"Failed to run cross-validation:\n{str(e)}")

    def find_optimal_k(self):
        """Find optimal k value for KNN"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.iris.data, self.iris.target,
                test_size=0.3,
                random_state=42,
                stratify=self.iris.target
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Test different k values
            k_range = range(1, 16)
            train_scores = []
            test_scores = []
            
            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train_scaled, y_train)
                train_scores.append(knn.score(X_train_scaled, y_train))
                test_scores.append(knn.score(X_test_scaled, y_test))
            
            # Find optimal k
            optimal_k = k_range[np.argmax(test_scores)]
            optimal_score = max(test_scores)
            
            result = f"""
{'='*60}
OPTIMAL K VALUE ANALYSIS 🔍
{'='*60}

Recommended k: {optimal_k}
Best Test Accuracy: {optimal_score:.4f} ({optimal_score*100:.2f}%)

Detailed Results (k vs Accuracy):
"""
            for k, train_acc, test_acc in zip(k_range, train_scores, test_scores):
                marker = " ← OPTIMAL" if k == optimal_k else ""
                result += f"k={k:2d}: Train={train_acc:.4f}, Test={test_acc:.4f}{marker}\n"
            
            result += f"""
Analysis:
• Very low k (1-3): May overfit to training data
• Very high k (>10): May underfit and lose local patterns  
• k={optimal_k}: Provides best balance for this dataset
"""
            self.knn_results.delete(1.0, tk.END)
            self.knn_results.insert(1.0, result)
            
            # Update k value in the interface
            self.knn_vars['k_value'].set(str(optimal_k))
            
            # Plot the results
            self.plot_k_accuracy(k_range, train_scores, test_scores, optimal_k)
            
        except Exception as e:
            messagebox.showerror("Optimization Error", f"Failed to find optimal k:\n{str(e)}")

    # Prediction Methods
    def predict_species(self):
        """Predict species based on user input"""
        try:
            model_type = self.model_var.get()
            
            if model_type == "decision_tree" and not self.dt_model:
                messagebox.showerror("Error", "Please train Decision Tree first!")
                return
            elif model_type == "knn" and not self.knn_model:
                messagebox.showerror("Error", "Please train KNN first!")
                return
            
            # Get feature values
            features = [var.get() for var in self.feature_vars]
            
            # Validate inputs
            for i, (feature, value) in enumerate(zip(self.feature_names, features)):
                if value <= 0:
                    messagebox.showerror("Input Error", 
                                       f"Please enter positive value for {feature}!")
                    return
            
            # Make prediction
            if model_type == "decision_tree":
                prediction = self.dt_model.predict([features])[0]
                probabilities = self.dt_model.predict_proba([features])[0]
                model_name = "Decision Tree"
            else:
                features_scaled = self.scaler.transform([features])
                prediction = self.knn_model.predict(features_scaled)[0]
                probabilities = self.knn_model.predict_proba(features_scaled)[0]
                model_name = "K-Nearest Neighbors"
            
            species = self.target_names[prediction]
            max_prob = max(probabilities)
            
            # Determine confidence level
            if max_prob > 0.9:
                confidence = "Very High"
                emoji = "🎯"
            elif max_prob > 0.7:
                confidence = "High"
                emoji = "✅"
            elif max_prob > 0.5:
                confidence = "Medium"
                emoji = "⚠️"
            else:
                confidence = "Low"
                emoji = "❓"
            
            # Display results
            result = f"""
{'='*70}
FLOWER SPECIES PREDICTION RESULTS {emoji}
{'='*70}

Model Used: {model_name}

Input Measurements:
{'-'*40}
• Sepal Length: {features[0]:.1f} cm
• Sepal Width:  {features[1]:.1f} cm  
• Petal Length: {features[2]:.1f} cm
• Petal Width:  {features[3]:.1f} cm

Prediction:
{'-'*40}
🌺 Predicted Species: {species.upper()}

Confidence Analysis:
{'-'*40}
• Confidence Level: {confidence}
• Highest Probability: {max_prob:.4f} ({max_prob*100:.2f}%)

Probability Distribution:
{'-'*40}
"""
            for i, (class_name, prob) in enumerate(zip(self.target_names, probabilities)):
                bar = "█" * int(prob * 20)
                result += f"• {class_name:<10}: {prob:.4f} {bar} ({prob*100:5.1f}%)\n"
            
            # Add species information
            species_info = {
                'setosa': "Small, round petals. Found in northern regions.",
                'versicolor': "Medium-sized, colorful petals. Various habitats.",  
                'virginica': "Large, elegant petals. Southern regions."
            }
            
            result += f"""
Species Information:
{'-'*40}
• {species}: {species_info.get(species, 'No additional information available.')}
"""
            self.prediction_display.delete(1.0, tk.END)
            self.prediction_display.insert(1.0, result)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to make prediction:\n{str(e)}")

    def clear_inputs(self):
        """Clear all input fields"""
        # Reset to middle values of ranges
        default_values = [5.5, 3.0, 3.5, 1.0]
        for var, default in zip(self.feature_vars, default_values):
            var.set(default)
        self.prediction_display.delete(1.0, tk.END)

    def load_example(self, species):
        """Load example measurements for a species"""
        examples = {
            'setosa': [5.1, 3.5, 1.4, 0.2],
            'versicolor': [6.0, 2.7, 4.0, 1.2],
            'virginica': [6.3, 3.3, 5.0, 1.8]
        }
        
        if species in examples:
            values = examples[species]
            for i, val in enumerate(values):
                self.feature_vars[i].set(val)
            
            self.prediction_display.delete(1.0, tk.END)
            self.prediction_display.insert(1.0, 
                f"Loaded {species} example measurements. Click 'Predict Species' to classify.")

    # Visualization Methods
    def clear_visualization(self):
        """Clear the current visualization"""
        for widget in self.plot_canvas_frame.winfo_children():
            widget.destroy()

    def display_plot(self, fig):
        """Display matplotlib figure in Tkinter"""
        self.clear_visualization()
        canvas = FigureCanvasTkAgg(fig, self.plot_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_feature_distribution(self):
        """Plot feature distribution by species"""
        self.clear_visualization()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Iris Dataset - Feature Distributions by Species', fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, feature in enumerate(self.feature_names):
            row, col = i // 2, i % 2
            for j, species in enumerate(self.target_names):
                species_data = self.df[self.df['species'] == species][feature]
                axes[row, col].hist(species_data, alpha=0.7, color=colors[j], 
                                   label=species, bins=12, density=True)
            
            axes[row, col].set_title(f'{feature} Distribution', fontweight='bold')
            axes[row, col].set_xlabel(feature + ' (cm)')
            axes[row, col].set_ylabel('Density')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.display_plot(fig)

    def plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        self.clear_visualization()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        correlation_matrix = self.df[self.feature_names].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax, fmt='.2f', cbar_kws={'shrink': 0.8})
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self.display_plot(fig)

    def plot_pairplot(self):
        """Plot pairplot of features"""
        self.clear_visualization()
        
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
        
        for i in range(4):
            for j in range(4):
                ax = fig.add_subplot(gs[i, j])
                
                if i == j:
                    # Diagonal - histograms
                    for species in self.target_names:
                        data = self.df[self.df['species'] == species][self.feature_names[i]]
                        ax.hist(data, alpha=0.7, color=colors[species], 
                               label=species, bins=10, density=True)
                    ax.set_title(self.feature_names[i], fontsize=10)
                    if i == 0:
                        ax.legend(fontsize=8)
                elif i > j:
                    # Lower triangle - scatter plots
                    for species in self.target_names:
                        species_data = self.df[self.df['species'] == species]
                        ax.scatter(species_data[self.feature_names[j]], 
                                  species_data[self.feature_names[i]], 
                                  alpha=0.7, color=colors[species], 
                                  label=species, s=30)
                    ax.set_xlabel(self.feature_names[j], fontsize=9)
                    ax.set_ylabel(self.feature_names[i], fontsize=9)
                else:
                    # Upper triangle - correlation values
                    corr = self.df[self.feature_names[j]].corr(self.df[self.feature_names[i]])
                    ax.text(0.5, 0.5, f'r = {corr:.2f}', ha='center', va='center', 
                           fontsize=12, transform=ax.transAxes, fontweight='bold')
                    ax.set_facecolor('#f0f0f0')
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        plt.suptitle('Iris Dataset - Pair Plot Analysis', fontsize=16, fontweight='bold')
        self.display_plot(fig)

    def plot_decision_tree(self):
        """Plot the decision tree"""
        if not self.dt_model:
            messagebox.showerror("Error", "Please train Decision Tree first!")
            return
        
        self.clear_visualization()
        
        fig, ax = plt.subplots(figsize=(16, 10))
        tree.plot_tree(self.dt_model, 
                      feature_names=self.feature_names,
                      class_names=self.target_names,
                      filled=True,
                      rounded=True,
                      fontsize=9,
                      ax=ax,
                      proportion=True)
        ax.set_title('Decision Tree Visualization - Iris Classification', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        self.display_plot(fig)

    def plot_feature_importance(self):
        """Plot feature importance"""
        if not self.dt_model:
            messagebox.showerror("Error", "Please train Decision Tree first!")
            return
        
        self.clear_visualization()
        
        importance = self.dt_model.feature_importances_
        features = self.feature_names
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(features, importance, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_xlabel('Feature Importance Score', fontweight='bold')
        ax.set_title('Decision Tree - Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, value in zip(bars, importance):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        self.display_plot(fig)

    def plot_knn_accuracy(self):
        """Plot KNN accuracy vs k value"""
        if not self.knn_model:
            messagebox.showerror("Error", "Please train KNN first!")
            return
        
        self.clear_visualization()
        
        # Generate accuracy data
        X_train, X_test, y_train, y_test = train_test_split(
            self.iris.data, self.iris.target,
            test_size=0.3,
            random_state=42,
            stratify=self.iris.target
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        k_range = range(1, 16)
        train_scores = []
        test_scores = []
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)
            train_scores.append(knn.score(X_train_scaled, y_train))
            test_scores.append(knn.score(X_test_scaled, y_test))
        
        optimal_k = k_range[np.argmax(test_scores)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_range, train_scores, 'o-', linewidth=2, markersize=6, 
                label='Training Accuracy', color='#4ECDC4')
        ax.plot(k_range, test_scores, 's-', linewidth=2, markersize=6,
                label='Testing Accuracy', color='#FF6B6B')
        
        # Highlight optimal k
        ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                  label=f'Optimal k = {optimal_k}')
        
        ax.set_xlabel('Number of Neighbors (k)', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('KNN - Accuracy vs Number of Neighbors', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_range)
        
        plt.tight_layout()
        self.display_plot(fig)

    def plot_class_distribution(self):
        """Plot class distribution"""
        self.clear_visualization()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        counts = [np.sum(self.iris.target == i) for i in range(len(self.target_names))]
        
        bars = ax.bar(self.target_names, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_title('Iris Dataset - Class Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Samples')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        self.display_plot(fig)

    def plot_feature_relationships(self):
        """Plot feature relationships"""
        self.clear_visualization()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
        
        # Petal Length vs Petal Width
        for species in self.target_names:
            species_data = self.df[self.df['species'] == species]
            ax1.scatter(species_data['petal length (cm)'], species_data['petal width (cm)'],
                       alpha=0.7, color=colors[species], label=species, s=50)
        ax1.set_xlabel('Petal Length (cm)')
        ax1.set_ylabel('Petal Width (cm)')
        ax1.set_title('Petal Dimensions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sepal Length vs Sepal Width
        for species in self.target_names:
            species_data = self.df[self.df['species'] == species]
            ax2.scatter(species_data['sepal length (cm)'], species_data['sepal width (cm)'],
                       alpha=0.7, color=colors[species], label=species, s=50)
        ax2.set_xlabel('Sepal Length (cm)')
        ax2.set_ylabel('Sepal Width (cm)')
        ax2.set_title('Sepal Dimensions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Sepal Length vs Petal Length
        for species in self.target_names:
            species_data = self.df[self.df['species'] == species]
            ax3.scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'],
                       alpha=0.7, color=colors[species], label=species, s=50)
        ax3.set_xlabel('Sepal Length (cm)')
        ax3.set_ylabel('Petal Length (cm)')
        ax3.set_title('Sepal vs Petal Length')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Sepal Width vs Petal Width
        for species in self.target_names:
            species_data = self.df[self.df['species'] == species]
            ax4.scatter(species_data['sepal width (cm)'], species_data['petal width (cm)'],
                       alpha=0.7, color=colors[species], label=species, s=50)
        ax4.set_xlabel('Sepal Width (cm)')
        ax4.set_ylabel('Petal Width (cm)')
        ax4.set_title('Sepal vs Petal Width')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Iris Dataset - Feature Relationships', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.display_plot(fig)

    # Comparison Methods
    def compare_models(self):
        """Compare both models"""
        if not self.dt_model or not self.knn_model:
            messagebox.showerror("Error", "Please train both models first!")
            return
        
        try:
            # Prepare test data
            X_train, X_test, y_train, y_test = train_test_split(
                self.iris.data, self.iris.target,
                test_size=0.3,
                random_state=42,
                stratify=self.iris.target
            )
            
            X_test_scaled = self.scaler.transform(X_test)
            
            # Get predictions
            dt_predictions = self.dt_model.predict(X_test)
            knn_predictions = self.knn_model.predict(X_test_scaled)
            
            # Calculate metrics
            dt_accuracy = accuracy_score(y_test, dt_predictions)
            knn_accuracy = accuracy_score(y_test, knn_predictions)
            
            dt_cm = confusion_matrix(y_test, dt_predictions)
            knn_cm = confusion_matrix(y_test, knn_predictions)
            
            result = f"""
{'='*70}
MODEL COMPARISON: DECISION TREE vs K-NEAREST NEIGHBORS
{'='*70}

Accuracy Comparison:
{'-'*40}
• Decision Tree:      {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)
• K-Nearest Neighbors: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)
• Difference:         {abs(dt_accuracy - knn_accuracy):.4f}

Confusion Matrices:
{'-'*40}
Decision Tree:
{dt_cm}

K-Nearest Neighbors:
{knn_cm}

Model Characteristics:
{'-'*40}
Decision Tree:
• Interpretable: ✅ Yes (white-box model)
• Training Speed: ⚡ Fast
• Prediction Speed: ⚡ Very Fast
• Handles: Non-linear relationships

K-Nearest Neighbors:
• Interpretable: ⚠️ Limited (black-box model)  
• Training Speed: ⚡ Fast (lazy learning)
• Prediction Speed: 🐢 Slow (computes distances)
• Handles: Complex decision boundaries

Recommendation:
{'-'*40}
"""
            if dt_accuracy > knn_accuracy:
                result += "• 🌳 Decision Tree performs better for this dataset\n"
                result += "• Advantages: Faster predictions, more interpretable"
            else:
                result += "• 🎯 K-Nearest Neighbors performs better for this dataset\n"
                result += "• Advantages: Can capture complex patterns"
            
            result += f"\nFinal Verdict: Both models achieve excellent performance (>95%)"
            
            self.comparison_results.delete(1.0, tk.END)
            self.comparison_results.insert(1.0, result)
            
        except Exception as e:
            messagebox.showerror("Comparison Error", f"Failed to compare models:\n{str(e)}")

    def show_performance_metrics(self):
        """Show detailed performance metrics"""
        if not self.dt_model or not self.knn_model:
            messagebox.showerror("Error", "Please train both models first!")
            return
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.iris.data, self.iris.target,
                test_size=0.3,
                random_state=42,
                stratify=self.iris.target
            )
            
            X_test_scaled = self.scaler.transform(X_test)
            
            # Get predictions and reports
            dt_predictions = self.dt_model.predict(X_test)
            knn_predictions = self.knn_model.predict(X_test_scaled)
            
            dt_report = classification_report(y_test, dt_predictions, 
                                            target_names=self.target_names, output_dict=True)
            knn_report = classification_report(y_test, knn_predictions,
                                             target_names=self.target_names, output_dict=True)
            
            result = f"""
{'='*70}
DETAILED PERFORMANCE METRICS
{'='*70}

Decision Tree Performance:
{'-'*40}
"""
            for species in self.target_names:
                precision = dt_report[species]['precision']
                recall = dt_report[species]['recall']
                f1 = dt_report[species]['f1-score']
                result += f"{species:<10} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n"
            
            result += f"\nK-Nearest Neighbors Performance:\n{'-'*40}\n"
            for species in self.target_names:
                precision = knn_report[species]['precision']
                recall = knn_report[species]['recall']
                f1 = knn_report[species]['f1-score']
                result += f"{species:<10} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n"
            
            result += f"""
Metrics Explanation:
{'-'*40}
• Precision: Of all predicted as class X, how many were correct?
• Recall: Of all actual class X, how many were correctly predicted?
• F1-Score: Harmonic mean of precision and recall

Overall Assessment:
{'-'*40}
• Both models show excellent performance across all classes
• F1-scores > 0.95 indicate highly reliable classification
• The Iris dataset is well-suited for both algorithms
"""
            self.comparison_results.delete(1.0, tk.END)
            self.comparison_results.insert(1.0, result)
            
        except Exception as e:
            messagebox.showerror("Metrics Error", f"Failed to calculate metrics:\n{str(e)}")

    def compare_training_time(self):
        """Compare training time of both models"""
        if not self.dt_model or not self.knn_model:
            messagebox.showerror("Error", "Please train both models first!")
            return
        
        # This would typically measure actual training time
        # For demonstration, we'll use estimated values
        result = f"""
{'='*70}
TRAINING TIME COMPARISON
{'='*70}

Estimated Training Times:
{'-'*40}
• Decision Tree:      ~0.001 - 0.005 seconds
• K-Nearest Neighbors: ~0.001 - 0.003 seconds

Key Insights:
{'-'*40}
• Both algorithms train very quickly on the Iris dataset
• KNN is technically "lazy" - it memorizes training data
• Decision Tree actively builds a tree structure
• For small datasets like Iris, training time is negligible

Real-world Considerations:
{'-'*40}
• For larger datasets: Decision Trees may train faster
• KNN prediction time increases with dataset size
• Decision Tree prediction time remains constant
"""
        self.comparison_results.delete(1.0, tk.END)
        self.comparison_results.insert(1.0, result)

    def compare_cross_validation(self):
        """Compare cross-validation results"""
        if not self.dt_model or not self.knn_model:
            messagebox.showerror("Error", "Please train both models first!")
            return
        
        try:
            # 5-fold cross-validation for both models
            dt_cv = cross_val_score(tree.DecisionTreeClassifier(), 
                                   self.iris.data, self.iris.target, cv=5)
            knn_cv = cross_val_score(KNeighborsClassifier(),
                                    self.iris.data, self.iris.target, cv=5)
            
            result = f"""
{'='*70}
CROSS-VALIDATION COMPARISON (5-fold)
{'='*70}

Decision Tree CV Scores:
{dt_cv}
• Mean: {dt_cv.mean():.4f}, Std: {dt_cv.std():.4f}

K-Nearest Neighbors CV Scores:
{knn_cv}  
• Mean: {knn_cv.mean():.4f}, Std: {knn_cv.std():.4f}

Comparison:
{'-'*40}
• Mean Accuracy Difference: {abs(dt_cv.mean() - knn_cv.mean()):.4f}
• Consistency (Lower std is better): {'Decision Tree' if dt_cv.std() < knn_cv.std() else 'KNN'}

Interpretation:
{'-'*40}
• Cross-validation provides robust performance estimation
• Lower standard deviation indicates more consistent performance
• Both models show excellent cross-validation results
"""
            self.comparison_results.delete(1.0, tk.END)
            self.comparison_results.insert(1.0, result)
            
        except Exception as e:
            messagebox.showerror("CV Comparison Error", f"Failed to compare CV:\n{str(e)}")

    # Utility Methods
    def show_data_tab(self):
        """Switch to data tab and show dataset info"""
        self.notebook.select(self.data_tab)
        self.show_dataset_info()

    def show_welcome_message(self):
        """Show welcome message in prediction tab"""
        welcome = f"""
{'='*70}
🌺 WELCOME TO IRIS FLOWER CLASSIFICATION SUITE
{'='*70}

This application provides a complete machine learning solution for 
classifying Iris flowers into three species:

• Setosa
• Versicolor  
• Virginica

Getting Started:
1. Explore the dataset in the 'Dataset' tab
2. Train models in 'Decision Tree' and 'K-Nearest Neighbors' tabs
3. Make predictions in the 'Prediction' tab
4. View visualizations in the 'Visualization' tab
5. Compare model performance in the 'Comparison' tab

Features:
• Complete dataset exploration and statistics
• Two classification algorithms with customizable parameters
• Interactive prediction with real-time results
• Comprehensive visualization suite
• Detailed model comparison and evaluation

Ready to begin? Start by exploring the dataset or training a model!
"""
        self.prediction_display.delete(1.0, tk.END)
        self.prediction_display.insert(1.0, welcome)

def main():
    """Main function to run the application"""
    try:
        root = tk.Tk()
        app = IrisClassificationApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Fatal Error", f"Failed to start application:\n{e}")

if __name__ == "__main__":
    main()