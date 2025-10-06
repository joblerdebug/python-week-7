"""
Iris Dataset Analysis
Complete data analysis and visualization of the Iris dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

class IrisAnalysis:
    def __init__(self):
        self.data = None
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and prepare the Iris dataset"""
        try:
            # Load dataset from sklearn
            iris = load_iris()
            self.data = iris
            
            # Create DataFrame
            self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
            self.df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
            
            # Save to CSV for reference
            self.df.to_csv('data/iris.csv', index=False)
            print("✅ Dataset loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
    
    def explore_data(self):
        """Explore dataset structure and basic information"""
        print("=" * 50)
        print("DATASET EXPLORATION")
        print("=" * 50)
        
        # Basic information
        print("\n📊 Dataset Shape:", self.df.shape)
        print("\n🔍 First 5 rows:")
        print(self.df.head())
        
        print("\n📝 Dataset Info:")
        print(self.df.info())
        
        print("\n❓ Missing Values:")
        print(self.df.isnull().sum())
        
        print("\n🎯 Data Types:")
        print(self.df.dtypes)
    
    def basic_analysis(self):
        """Perform basic statistical analysis"""
        print("\n" + "=" * 50)
        print("BASIC STATISTICAL ANALYSIS")
        print("=" * 50)
        
        # Basic statistics
        print("\n📈 Descriptive Statistics:")
        print(self.df.describe())
        
        # Species-wise analysis
        print("\n🌷 Species-wise Analysis:")
        species_stats = self.df.groupby('species').agg({
            'sepal length (cm)': ['mean', 'std', 'min', 'max'],
            'sepal width (cm)': ['mean', 'std', 'min', 'max'],
            'petal length (cm)': ['mean', 'std', 'min', 'max'],
            'petal width (cm)': ['mean', 'std', 'min', 'max']
        })
        print(species_stats)
        
        # Correlation analysis
        print("\n🔗 Correlation Matrix:")
        numeric_df = self.df.select_dtypes(include=[np.number])
        print(numeric_df.corr())
    
    def create_visualizations(self):
        """Create all required visualizations"""
        print("\n" + "=" * 50)
        print("DATA VISUALIZATIONS")
        print("=" * 50)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Iris Dataset Analysis - Comprehensive Visualizations', fontsize=16, fontweight='bold')
        
        # 1. Line Chart - Feature trends by sample index
        self._create_line_chart(axes[0, 0])
        
        # 2. Bar Chart - Average measurements by species
        self._create_bar_chart(axes[0, 1])
        
        # 3. Histogram - Distribution of sepal length
        self._create_histogram(axes[1, 0])
        
        # 4. Scatter Plot - Sepal length vs Petal length
        self._create_scatter_plot(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('iris_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional: Correlation heatmap
        self._create_heatmap()
        
        # Additional: Box plots
        self._create_box_plots()
    
    def _create_line_chart(self, ax):
        """Line chart showing feature trends"""
        features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        
        for feature in features:
            ax.plot(self.df[feature].values, label=feature, alpha=0.7)
        
        ax.set_title('Line Chart: Feature Trends Across Samples', fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Measurement (cm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_bar_chart(self, ax):
        """Bar chart comparing average measurements by species"""
        species_means = self.df.groupby('species').mean()
        
        x = np.arange(len(species_means.index))
        width = 0.2
        
        features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, feature in enumerate(features):
            ax.bar(x + i*width, species_means[feature], width, label=feature, color=colors[i], alpha=0.8)
        
        ax.set_title('Bar Chart: Average Measurements by Species', fontweight='bold')
        ax.set_xlabel('Species')
        ax.set_ylabel('Average Measurement (cm)')
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(species_means.index, rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _create_histogram(self, ax):
        """Histogram of sepal length distribution"""
        species = self.df['species'].unique()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, specie in enumerate(species):
            ax.hist(self.df[self.df['species'] == specie]['sepal length (cm)'], 
                   alpha=0.7, label=specie, color=colors[i], bins=15)
        
        ax.set_title('Histogram: Sepal Length Distribution by Species', fontweight='bold')
        ax.set_xlabel('Sepal Length (cm)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_scatter_plot(self, ax):
        """Scatter plot of sepal length vs petal length"""
        species = self.df['species'].unique()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        markers = ['o', 's', '^']
        
        for i, specie in enumerate(species):
            specie_data = self.df[self.df['species'] == specie]
            ax.scatter(specie_data['sepal length (cm)'], specie_data['petal length (cm)'],
                      c=colors[i], marker=markers[i], label=specie, alpha=0.7, s=60)
        
        ax.set_title('Scatter Plot: Sepal Length vs Petal Length', fontweight='bold')
        ax.set_xlabel('Sepal Length (cm)')
        ax.set_ylabel('Petal Length (cm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_heatmap(self):
        """Create correlation heatmap"""
        plt.figure(figsize=(8, 6))
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Heatmap: Feature Correlation Matrix', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('iris_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_box_plots(self):
        """Create box plots for all features"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        
        for i, feature in enumerate(features):
            ax = axes[i//2, i%2]
            self.df.boxplot(column=feature, by='species', ax=ax)
            ax.set_title(f'Box Plot: {feature}', fontweight='bold')
            ax.set_ylabel('Measurement (cm)')
        
        fig.suptitle('')
        plt.tight_layout()
        plt.savefig('iris_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def summarize_findings(self):
        """Summarize key findings from the analysis"""
        print("\n" + "=" * 50)
        print("KEY FINDINGS AND OBSERVATIONS")
        print("=" * 50)
        
        print("\n🔍 **Key Insights:**")
        print("1. Setosa species has distinctly smaller petal measurements")
        print("2. Virginica generally has the largest measurements across all features")
        print("3. Strong positive correlation between petal length and petal width")
        print("4. Sepal width shows the least variation across species")
        print("5. Clear separation between species in scatter plots")
        
        print("\n📊 **Dataset Quality:**")
        print(f"- No missing values: {not self.df.isnull().any().any()}")
        print(f"- Balanced classes: {self.df['species'].value_counts().to_dict()}")
        print(f"- Consistent measurements across 150 samples")
        
        print("\n🎯 **Potential Applications:**")
        print("- Species classification model training")
        print("- Pattern recognition in botanical measurements")
        print("- Educational dataset for machine learning")

def main():
    """Main execution function"""
    print("🌷 IRIS DATASET ANALYSIS")
    print("=" * 50)
    
    # Initialize analysis
    analyzer = IrisAnalysis()
    
    # Perform all analysis steps
    analyzer.explore_data()
    analyzer.basic_analysis()
    analyzer.create_visualizations()
    analyzer.summarize_findings()
    
    print("\n✅ Analysis completed successfully!")
    print("📁 Outputs saved:")
    print("   - iris_analysis.png (main visualizations)")
    print("   - iris_correlation.png (correlation heatmap)")
    print("   - iris_boxplots.png (box plots)")
    print("   - data/iris.csv (dataset)")

if __name__ == "__main__":
    main()
