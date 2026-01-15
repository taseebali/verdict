"""Demo dataset generator."""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_business_dataset():
    """Generate a demo business dataset for testing."""
    np.random.seed(42)
    
    n_samples = 500
    
    # Create features
    data = {
        "order_value": np.random.exponential(50, n_samples) + 10,
        "customer_age": np.random.randint(18, 75, n_samples),
        "days_since_signup": np.random.randint(1, 365, n_samples),
        "num_previous_orders": np.random.randint(0, 50, n_samples),
        "customer_region": np.random.choice(["North", "South", "East", "West"], n_samples),
        "product_category": np.random.choice(["Electronics", "Clothing", "Home", "Sports"], n_samples),
        "payment_method": np.random.choice(["Credit Card", "Debit Card", "PayPal"], n_samples),
        "time_on_site_minutes": np.random.exponential(10, n_samples) + 1,
        "items_in_cart": np.random.randint(1, 10, n_samples),
        "discount_applied": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on features (churn/conversion prediction)
    # Higher order value and more previous orders = less likely to churn
    churn_prob = (0.3 
                  - 0.01 * (df["order_value"] - 10) / 100  # Negative correlation with order value
                  - 0.02 * df["num_previous_orders"] / 10   # Negative correlation with order history
                  - 0.001 * df["customer_age"] / 10          # Slight negative correlation with age
                  + 0.05 * (1 - df["discount_applied"]))    # Positive correlation with no discount
    
    churn_prob = np.clip(churn_prob, 0, 1)
    df["churn"] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    return df

if __name__ == "__main__":
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    df = generate_business_dataset()
    output_path = os.path.join(output_dir, "demo_business_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Demo dataset generated: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nTarget distribution:\n{df['churn'].value_counts()}")
