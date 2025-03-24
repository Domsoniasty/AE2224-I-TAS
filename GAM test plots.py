from pygam import LogisticGAM, s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example data
df = pd.DataFrame({
    'temperature': np.random.uniform(-10, 30, 200),  # Continuous variable
    'wind_speed': np.random.uniform(0, 20, 200),      # Continuous variable
    'gravity_wave': np.random.choice([0, 1], 200)     # Binary outcome
})

# Fit a GAM
gam = LogisticGAM(s(0) + s(1)).fit(df[['temperature', 'wind_speed']].values, df['gravity_wave'].values)


# Plot the effect of each condition
plt.figure(figsize=(10, 5))

# Plot each smooth function
for i, feature in enumerate(['Temperature', 'Wind Speed']):
    plt.subplot(1, 2, i + 1)

    XX = gam.generate_X_grid(term=i)  # Generate points for plotting
    plt.plot(XX[:, i], gam.partial_dependence(term=i))  # Plot the smooth function
    plt.title(f'Effect of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Log-odds of Gravity Wave')


plt.tight_layout()
plt.show()