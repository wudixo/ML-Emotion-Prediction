import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Historical data: Year vs. Estimated global cost in trillion USD
years = np.array([2010, 2020, 2025]).reshape(-1, 1)
costs = np.array([2.5, 3.5, 4.5])  # Hypothetical past data for illustration

# Fit linear regression model
model = LinearRegression()
model.fit(years, costs)

# Predict costs up to 2030
future_years = np.array(range(2010, 2031)).reshape(-1, 1)
predicted_costs = model.predict(future_years)

# Plot the projection
plt.figure(figsize=(10,6))
plt.plot(future_years, predicted_costs, marker='o')
plt.title('Projected Global Cost of Mental Health Issues (2010-2030)')
plt.xlabel('Year')
plt.ylabel('Cost (Trillion USD)')
plt.grid(True)
plt.tight_layout()
plt.show()
