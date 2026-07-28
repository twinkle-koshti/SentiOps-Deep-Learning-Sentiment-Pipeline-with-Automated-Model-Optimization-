import os
import pandas as pd

# Data directory check karke create karna agar exist nahi karti
os.makedirs("data", exist_ok=True)

# Fake customer reviews sample dataset structure
mock_data = {
    "text": [
        "The customer service was absolutely fantastic and helpful!",
        "Horrible experience. The product arrived broken and completely unusable.",
        "It is okay, neither too good nor too bad. Average performance.",
        "Super fast local delivery, highly recommend this platform!",
        "Worst battery backup ever, my phone drains completely in just two hours."
    ],
    "user_id": [101, 102, 103, 104, 105] # Extra column just for structural testing
}

# Converting dictionary to dataframe
df = pd.DataFrame(mock_data)

# File save target injection
file_path = "data/sample_reviews.csv"
df.to_csv(file_path, index=False)

print(f"🎉 Success! Your testing dataset is ready at: {file_path}")
