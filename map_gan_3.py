import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from folium.plugins import HeatMap

# Load and preprocess your spatial data (assuming it's in a CSV file)
df = pd.read_csv('your_spatial_data.csv')
#

# Generate synthetic data
def generate_real_data(samples):
    factors = np.random.rand(samples, 5)  # 5 factors affecting cost of living
    cost_of_living = np.sum(factors, axis=1) + np.random.normal(0, 0.1, samples)
    return factors, cost_of_living

# Generator model
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(16, input_dim=latent_dim, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model


# Discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Dense(16, input_dim=1, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build and compile GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
    return gan
# Training loop
def train_gan(generator, discriminator, gan, real_data, epochs, batch_size):
    for epoch in range(epochs):
        for _ in range(batch_size):
            real_factors, real_cost_of_living = real_data
            real_labels = np.ones(batch_size)  # All real data is labeled as 1
            fake_cost_of_living = generator.predict(np.random.rand(batch_size, latent_dim))
            fake_labels = np.zeros(batch_size)
            
            d_loss_real = discriminator.train_on_batch(real_cost_of_living, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_cost_of_living, fake_labels)
            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            fake_labels_gan = np.ones(batch_size)
            g_loss = gan.train_on_batch(np.random.rand(batch_size, latent_dim), fake_labels_gan)
            
        print(f"Epoch {epoch+1}, D Loss: {d_loss}, G Loss: {g_loss}")

# Calculate factor importance
def calculate_factor_importance(generator, latent_dim):
    # Initialize an array to store the effects of each factor
    factor_effects = np.zeros(latent_dim)
    
    # Generate random noise
    noise = np.random.rand(1000, latent_dim)
    
    # Generate synthetic cost of living values
    generated_cost_of_living = generator.predict(noise)
    
    # Calculate the effect of each factor by subtracting the mean effect
    mean_effect = np.mean(generated_cost_of_living)
    
    # Calculate the effect of each factor individually
    for i in range(latent_dim):
        factor_noise = noise.copy()
        factor_noise[:, i] = 0  # Set one factor to zero while keeping others constant
        factor_effect = np.mean(generator.predict(factor_noise) - generated_cost_of_living)
        factor_effects[i] = factor_effect
    
    # Print the factor effects
    print("Factor Effects:")
    for i, effect in enumerate(factor_effects):
        print(f"Factor {i+1}: {effect}")

# Hyperparameters
latent_dim = 10  # Latent dimension for generator
epochs = 10
batch_size = 128

# Build models
generator = build_generator(latent_dim)
discriminator = build_discriminator()

# Compile the discriminator model
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Build GAN model and compile it
gan = build_gan(generator, discriminator)

# Training loop
real_data = generate_real_data(batch_size)
train_gan(generator, discriminator, gan, real_data, epochs, batch_size)

# Calculate factor importance
calculate_factor_importance(generator, latent_dim)

# Generate synthetic data
num_samples = 1000
generated_cost_of_living = generator.predict(np.random.rand(num_samples, latent_dim))
# Create a map centered around a specific location (e.g., a city)
m = folium.Map(location=[53.350140, -6.266155], zoom_start=12)

#54.26969 -8.46943

# Add a heatmap layer using your cost of living data
heat_data = [[row['latitude'], row['longitude'], row['cost_of_living']] for index, row in df.iterrows()]
folium.plugins.HeatMap(heat_data).add_to(m)

# Display the map
m.save('cost_of_living_heatmap.html')
# Visualize generated data
plt.scatter(np.sum(np.random.rand(num_samples, 5), axis=1), generated_cost_of_living, label='Generated Data')
plt.xlabel('Sum of Factors')
plt.ylabel('Cost of Living')
plt.legend()
plt.show()


