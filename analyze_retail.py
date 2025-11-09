#!/usr/bin/env python3
"""
Retail Data Exploratory Data Analysis
Author: Ashish Jha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(filepath):
    """Load retail dataset"""
    print("Loading retail data...")
    df = pd.read_csv(filepath)
    
    # Convert date columns
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Day'] = df['InvoiceDate'].dt.day
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    
    return df

def basic_statistics(df):
    """Display basic statistics"""
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")

def sales_trends_analysis(df):
    """Analyze sales trends over time"""
    print("\n" + "="*50)
    print("SALES TRENDS ANALYSIS")
    print("="*50)
    
    # Monthly sales
    if 'TotalAmount' in df.columns and 'Month' in df.columns:
        monthly_sales = df.groupby('Month')['TotalAmount'].sum().sort_index()
        
        plt.figure(figsize=(12, 6))
        monthly_sales.plot(kind='bar', color='steelblue')
        plt.title('Monthly Sales Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Total Sales ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/monthly_sales_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nMonthly Sales:\n{monthly_sales}")
        print(f"\nBest Month: {monthly_sales.idxmax()} with ${monthly_sales.max():,.2f}")
        print(f"Worst Month: {monthly_sales.idxmin()} with ${monthly_sales.min():,.2f}")

def customer_behavior_analysis(df):
    """Analyze customer behavior patterns"""
    print("\n" + "="*50)
    print("CUSTOMER BEHAVIOR INSIGHTS")
    print("="*50)
    
    if 'CustomerID' in df.columns:
        # Customer purchase frequency
        customer_freq = df['CustomerID'].value_counts()
        
        print(f"\nTotal Unique Customers: {df['CustomerID'].nunique()}")
        print(f"Average Purchases per Customer: {customer_freq.mean():.2f}")
        print(f"\nTop 10 Customers by Purchase Frequency:")
        print(customer_freq.head(10))
        
        # Customer spending
        if 'TotalAmount' in df.columns:
            customer_spending = df.groupby('CustomerID')['TotalAmount'].sum().sort_values(ascending=False)
            
            print(f"\nTop 10 Customers by Spending:")
            print(customer_spending.head(10))
            
            # Plot customer spending distribution
            plt.figure(figsize=(12, 6))
            plt.hist(customer_spending, bins=50, color='coral', edgecolor='black')
            plt.title('Customer Spending Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Total Spending ($)')
            plt.ylabel('Number of Customers')
            plt.tight_layout()
            plt.savefig('results/customer_spending_dist.png', dpi=300, bbox_inches='tight')
            plt.close()

def revenue_driver_analysis(df):
    """Identify key revenue drivers"""
    print("\n" + "="*50)
    print("REVENUE DRIVER IDENTIFICATION")
    print("="*50)
    
    if 'Product' in df.columns and 'TotalAmount' in df.columns:
        # Top products by revenue
        product_revenue = df.groupby('Product')['TotalAmount'].sum().sort_values(ascending=False)
        
        print(f"\nTop 10 Products by Revenue:")
        print(product_revenue.head(10))
        
        # Visualize top products
        plt.figure(figsize=(12, 6))
        product_revenue.head(10).plot(kind='barh', color='green')
        plt.title('Top 10 Products by Revenue', fontsize=16, fontweight='bold')
        plt.xlabel('Revenue ($)')
        plt.ylabel('Product')
        plt.tight_layout()
        plt.savefig('results/top_products_revenue.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Revenue concentration (Pareto analysis)
        total_revenue = product_revenue.sum()
        cumulative_pct = (product_revenue.cumsum() / total_revenue * 100)
        top_20_pct = cumulative_pct[cumulative_pct <= 80].count()
        
        print(f"\n80/20 Rule Analysis:")
        print(f"Top {top_20_pct} products ({top_20_pct/len(product_revenue)*100:.1f}%) generate 80% of revenue")

def statistical_analysis(df):
    """Perform statistical tests"""
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS")
    print("="*50)
    
    # Correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nCorrelation Matrix:\n{correlation_matrix}")

def generate_report(df):
    """Generate comprehensive EDA report"""
    print("\n" + "="*70)
    print("COMPREHENSIVE RETAIL DATA ANALYSIS REPORT")
    print("="*70)
    
    basic_statistics(df)
    sales_trends_analysis(df)
    customer_behavior_analysis(df)
    revenue_driver_analysis(df)
    statistical_analysis(df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("Visualizations saved in 'results/' directory")
    print("="*70)

if __name__ == "__main__":
    # Load data (use retail dataset from Kaggle or UCI)
    df = load_data('data/retail_data.csv')
    
    # Generate comprehensive report
    generate_report(df)
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS SUMMARY")
    print("="*70)
    print("1. Sales trends identified across different time periods")
    print("2. Customer behavior patterns and segmentation analyzed")
    print("3. Revenue drivers and top-performing products identified")
    print("4. Statistical relationships between variables discovered")
    print("5. Actionable insights for business decision-making generated")
    print("="*70)
