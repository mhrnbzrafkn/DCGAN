import pandas as pd
import matplotlib.pyplot as plt

base_path = './training-src/output_results/output_report'

# Load discriminator loss data
discriminator_loss_df = pd.read_csv(f'{base_path}/discriminator_losses.csv')

# Load generator loss data
generator_loss_df = pd.read_csv(f'{base_path}/generator_losses.csv')

# Merge the data frames on 'Epoch' column
merged_df = pd.merge(discriminator_loss_df, generator_loss_df, on="Epoch")

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(merged_df['Epoch'], merged_df['Discriminator Loss'], label='Discriminator Loss')
plt.plot(merged_df['Epoch'], merged_df['Generator Loss'], label='Generator Loss')
plt.title('GAN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'{base_path}/loss_plot.png')
plt.show()
